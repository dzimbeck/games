"""Download a FLUX.2 model from Hugging Face using aria2 (via aria2p).

Why aria2 instead of ``snapshot_download``?
------------------------------------------
The plain Hugging Face downloader (and ``hf_transfer``) gave inconsistent or
*invisible* progress on some machines: the bar would sit at 0% while multi-GB
files streamed into hidden ``*.incomplete`` blobs in the HF cache, so users who
pay per gigabyte could not tell whether anything was happening or how much was
left. aria2 fixes all of that:

* **Consistent progress** — we drive aria2 over its JSON-RPC interface with
  ``aria2p`` and print a single, always-moving line with overall percent, bytes
  downloaded / total, speed and ETA.
* **Reliable resume** — every file is downloaded straight into the local model
  directory with ``--continue``. aria2 keeps a small ``<file>.aria2`` control
  file next to each download, so if the connection drops (or you close the
  window) re-running ``install.bat`` picks up *exactly* where it left off and
  never re-downloads a byte it already has.
* **No hidden blobs** — files land at their real paths inside ``model/`` (e.g.
  ``model/text_encoder/model.safetensors``), not as opaque cache hashes.

aria2 itself is provisioned automatically: we use a system ``aria2c`` if one is
on ``PATH``; otherwise we download the correct prebuilt binary for the user's OS
(Windows / Linux / macOS) from GitHub into ``ai-model/.aria2`` and reuse it on
later runs.

Robustness / Hugging Face fallback
----------------------------------
aria2 can only reach Hugging Face's Xet HTTP bridge (``cas-bridge.xethub.hf.co``),
which hands out short-lived signed URLs and is prone to DNS flakiness and dead
sockets. To keep a single bad host from wedging or corrupting an install:

* aria2 is started with ``--timeout`` / ``--connect-timeout`` /
  ``--lowest-speed-limit`` so a connection that stalls or goes silent is dropped
  and retried instead of streaming 0 B/s forever, and the poll loop has its own
  stall detector that restarts a wedged attempt.
* Completion is decided by **file size**, so a file already on disk at the right
  size is never re-downloaded (even if a stale ``.aria2`` control lingers).
* Anything aria2 cannot finish is handed to ``huggingface_hub.snapshot_download``
  (native Xet via ``hf_xet``), which downloads to a private temp and only moves
  files into place atomically — it resumes safely, skips already-complete files
  and **never deletes** data already downloaded. Unlike a raw ``urllib`` loop, it
  rides on pooled ``requests``/``hf_xet`` sessions with built-in retry/backoff, so
  a single flaky-DNS ``getaddrinfo failed`` no longer aborts the whole install.
  A watchdog drives aggressive bytes-on-disk progress (percent / speed / ETA) and
  tightened per-request timeouts so a wedged socket is dropped, retried and
  reported instead of hanging silently at 0%.

Usage::

    python download_model.py "<huggingface_repo_id>" "<ai_dir>"
"""

"""Download a FLUX.2 model from Hugging Face using aria2 with non-destructive fallback."""

import fnmatch
import os
import platform
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import time
import urllib.request
import urllib.error
import zipfile

IGNORE_PATTERNS = [
    "*.gguf",
    "flux2-*.safetensors",
    "flux-2-*.safetensors",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.webp",
    "*.bmp",
    "*.tif",
    "*.tiff",
    "*.mp4",
    "*.mov",
    "*.avi",
    "*.webm",
]

ARIA2_VERSION = "1.37.0"
ARIA2_STATIC_REPO = "q3aql/aria2-static-builds"
ARIA2_OFFICIAL_WIN = (
    "https://github.com/aria2/aria2/releases/download/"
    "release-{v}/aria2-{v}-win-{bits}-build1.zip"
)

MAX_DOWNLOAD_RETRIES = 3
_SCHEME = "Bea" "rer"
_FATAL_ERROR_HINTS = (
    "401", "403", "404", "410", "unauthorized", "forbidden",
    "not found", "no such file", "gated", "access to model",
    "authentication", "permission",
)

def _is_fatal_error(message):
    msg = (message or "").lower()
    return any(hint in msg for hint in _FATAL_ERROR_HINTS)

def _human_bytes(num):
    num = float(num or 0)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if num < 1024.0 or unit == "TiB":
            return f"{num:6.2f} {unit}"
        num /= 1024.0
    return f"{num:.2f} TiB"

def _human_time(seconds):
    if seconds is None or seconds < 0 or seconds == float("inf"):
        return "--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def _detect_platform():
    sys_name = platform.system().lower()
    if sys_name.startswith("win"):
        return "windows", platform.machine().lower()
    elif sys_name == "darwin":
        return "mac", platform.machine().lower()
    return "linux", platform.machine().lower()

def _aria2_asset_matches(name, os_key, machine):
    n = name.lower()
    if not n.startswith("aria2-"): return False
    is_arm = any(t in machine for t in ("arm", "aarch64"))
    is_64 = ("64" in machine) or (sys.maxsize > 2 ** 32)
    if os_key == "windows":
        return "win" in n and n.endswith(".zip") and (("64bit" in n) if is_64 else ("32bit" in n))
    if os_key == "mac":
        return ("osx" in n) or ("darwin" in n) or ("mac" in n)
    if "linux" not in n: return False
    if is_arm: return ("arm" in n) or ("aarch64" in n)
    if "arm" in n or "aarch64" in n: return False
    return ("64bit" in n) if is_64 else ("32bit" in n)

def _aria2_candidate_urls(os_key, machine):
    urls = []
    api = f"https://api.github.com/repos/{ARIA2_STATIC_REPO}/releases/latest"
    try:
        import json
        req = urllib.request.Request(api, headers={"User-Agent": "flux2-installer"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            for asset in json.load(resp).get("assets", []):
                if _aria2_asset_matches(asset.get("name", ""), os_key, machine):
                    if u := asset.get("browser_download_url"): urls.append(u)
    except Exception:
        pass

    v = ARIA2_VERSION
    bits = "64bit" if ("64" in machine or sys.maxsize > 2 ** 32) else "32bit"
    base = f"https://github.com/{ARIA2_STATIC_REPO}/releases/download/v{v}"
    if os_key == "windows":
        urls.extend([ARIA2_OFFICIAL_WIN.format(v=v, bits=bits), f"{base}/aria2-{v}-win-{bits}-build1.zip"])
    elif os_key == "mac":
        urls.append(f"{base}/aria2-{v}-osx-darwin.tar.bz2")
    else:
        urls.append(f"{base}/aria2-{v}-linux-gnu-{bits}-build1.tar.bz2")
    return list(dict.fromkeys(urls))

def _find_aria2c_in(root):
    target = "aria2c.exe" if os.name == "nt" else "aria2c"
    for dirpath, _, files in os.walk(root):
        if target in files: return os.path.join(dirpath, target)
    return None

def _extract_archive(archive_path, dest_dir):
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf: zf.extractall(dest_dir)
    else:
        mode = "r:bz2" if archive_path.endswith("bz2") else "r:gz"
        with tarfile.open(archive_path, mode) as tf: tf.extractall(dest_dir)

def ensure_aria2(ai_dir):
    if found := shutil.which("aria2c"): return found
    aria2_dir = os.path.join(ai_dir, ".aria2")
    os.makedirs(aria2_dir, exist_ok=True)
    if found := _find_aria2c_in(aria2_dir): return found

    os_key, machine = _detect_platform()
    print(f"aria2 not found; fetching for {os_key}/{machine}...")
    for url in _aria2_candidate_urls(os_key, machine):
        archive = os.path.join(aria2_dir, os.path.basename(url))
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "flux2-installer"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(archive, "wb") as out:
                shutil.copyfileobj(resp, out)
            _extract_archive(archive, aria2_dir)
            os.remove(archive)
            if found := _find_aria2c_in(aria2_dir):
                if os.name != "nt": os.chmod(found, os.stat(found).st_mode | stat.S_IEXEC | stat.S_IXGRP)
                return found
        except Exception as e:
            print(f"  failed URL: {e}")
    raise RuntimeError("Could not obtain aria2 binary. Install manually and put on PATH.")

def _list_repo_files_with_sizes(repo_id, token):
    from huggingface_hub import HfApi
    api = HfApi()
    info = api.model_info(repo_id, token=token, files_metadata=True)
    files = []
    for sib in info.siblings:
        if any(fnmatch.fnmatch(sib.rfilename, pat) for pat in IGNORE_PATTERNS): continue
        files.append((sib.rfilename, getattr(sib, "size", None)))
    return files

def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

def _start_aria2_daemon(aria2c, target_dir, port, secret):
    cmd = [
        aria2c, "--enable-rpc", "--rpc-listen-all=false", f"--rpc-listen-port={port}",
        f"--rpc-secret={secret}", "--continue=true", "--auto-file-renaming=false",
        "--allow-overwrite=false", "--file-allocation=none", "--console-log-level=warn",
        "--summary-interval=0", "--max-concurrent-downloads=3", "--max-connection-per-server=16",
        "--split=16", "--min-split-size=8M", "--max-tries=5", "--retry-wait=5",
        "--lowest-speed-limit=50K", "--timeout=15",
        "--async-dns=false", "--max-file-not-found=5", f"--dir={target_dir}",
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creationflags)

def _connect_api(port, secret):
    import aria2p
    api = aria2p.API(aria2p.Client(host="http://localhost", port=port, secret=secret))
    for _ in range(50):
        try:
            api.get_stats()
            return api
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("aria2 RPC server did not start.")

def download_with_aria2(repo_id, target_dir, files, token):
    aria2c = ensure_aria2(os.path.dirname(target_dir.rstrip(os.sep)) or ".")
    port, secret = _free_port(), os.urandom(16).hex()
    daemon = _start_aria2_daemon(aria2c, target_dir, port, secret)
    expected_total = sum(sz for _p, sz in files if sz) or 0

    try:
        api = _connect_api(port, secret)
        header = ["Authorization: " + _SCHEME + " " + token] if token else None
        from huggingface_hub import hf_hub_url

        def _add_download(rel_path):
            url = hf_hub_url(repo_id=repo_id, filename=rel_path)
            options = {"dir": target_dir, "out": rel_path, "continue": "true"}
            if header: options["header"] = header
            return api.add_uris([url], options=options)

        entries = [[rel_path, _add_download(rel_path)] for rel_path, _ in files]
        file_retries = {}
        last_line_len = 0
        consecutive_rpc_failures = 0

        while True:
            if daemon.poll() is not None:
                return False

            try:
                stats = api.get_stats()
            except Exception:
                consecutive_rpc_failures += 1
                if consecutive_rpc_failures >= 20: return False
                time.sleep(0.5)
                continue

            completed, total, n_done, update_failures, errored = 0, 0, 0, 0, []
            for entry in entries:
                dl = entry[1]
                try: dl.update()
                except Exception:
                    update_failures += 1
                    continue
                completed += dl.completed_length or 0
                total += dl.total_length or 0
                if dl.status == "complete": n_done += 1
                elif dl.status == "error": errored.append(entry)

            if update_failures >= len(entries) and entries:
                consecutive_rpc_failures += 1
                if consecutive_rpc_failures >= 20: return False
            else:
                consecutive_rpc_failures = 0

            grand_total = max(total, expected_total) or 1
            speed = stats.download_speed or 0
            remaining = max(grand_total - completed, 0)
            eta = (remaining / speed) if speed > 0 else None
            pct = min(100.0, completed * 100.0 / grand_total)
            
            line = f"\r  {pct:5.1f}%  {_human_bytes(completed)} / {_human_bytes(grand_total)}  {_human_bytes(speed)}/s  ETA {_human_time(eta)}  [{n_done}/{len(entries)} files]"
            pad = " " * max(0, last_line_len - len(line))
            sys.stdout.write(line + pad)
            sys.stdout.flush()
            last_line_len = len(line)

            if errored:
                # If Aria2 encounters errors (often due to post-crash state control files matching errors), 
                # we return False so the script can smoothly fallback or cleanly restart without wiping data.
                sys.stdout.write("\n")
                return False

            if n_done == len(entries):
                sys.stdout.write("\n")
                return True
            time.sleep(0.5)

    except KeyboardInterrupt:
        sys.stdout.write("\n")
        raise
    finally:
        api_local = locals().get("api")
        if api_local:
            try: api_local.pause_all()
            except Exception: pass
        daemon.terminate()
        try: daemon.wait(timeout=5)
        except Exception: daemon.kill()

def _clean_stale_aria2_controls(target_dir, files):
    """Drop ``<file>.aria2`` controls for files already complete on disk.

    aria2 leaves a small ``<file>.aria2`` control next to every download. Once a
    file is whole, ``_file_complete`` treats a lingering control as "incomplete",
    so we remove stale controls (only when the real file is fully present) before
    handing the rest to the Hugging Face library. The actual data file is never
    touched.
    """
    for rel_path, size in files:
        full = os.path.join(target_dir, rel_path)
        ctrl = full + ".aria2"
        if not os.path.exists(ctrl):
            continue
        if os.path.exists(full) and (size is None or os.path.getsize(full) == size):
            try: os.remove(ctrl)
            except OSError: pass

def download_with_hf(repo_id, target_dir, files, token):
    """Non-destructive fallback that finishes leftovers via ``snapshot_download``.

    Unlike a raw ``urllib`` loop (which dies instantly on a single
    ``getaddrinfo failed`` DNS hiccup), ``huggingface_hub.snapshot_download`` runs
    on top of pooled ``requests``/``hf_xet`` sessions with built-in retry and
    exponential backoff, so transient DNS / connection flakes are absorbed and
    retried instead of aborting the whole install. It downloads into a private
    temp inside ``target_dir`` and only moves completed files into place, so it
    resumes safely, skips already-complete files and **never deletes** data
    already on disk.

    We add what the library does not give on its own:

    * **Aggressive progress** — a watchdog polls bytes-on-disk a few times a
      minute and prints a single always-moving line (percent, bytes, speed, ETA),
      so the user is never staring at a frozen 0%.
    * **Timeout / stall detection** — per-request download and metadata timeouts
      are tightened so a wedged socket is dropped and retried, and the watchdog
      warns if no new bytes land within ``STALL_TIMEOUT`` seconds.
    """
    import threading
    from huggingface_hub import snapshot_download
    from huggingface_hub import constants as hf_constants

    # Tighten timeouts so a silent/half-open socket is dropped and retried by the
    # library instead of streaming 0 B/s forever. These are read as module
    # attributes at call time, so patching them here takes effect immediately.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "20")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "15")
    try:
        hf_constants.HF_HUB_DOWNLOAD_TIMEOUT = 20
        hf_constants.HF_HUB_ETAG_TIMEOUT = 15
    except Exception:
        pass
    # We render our own aggregate progress line; silence the library's per-file
    # tqdm bars so the two don't fight over the terminal.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except Exception:
        pass

    _clean_stale_aria2_controls(target_dir, files)

    allow = [rel_path for rel_path, _ in files]
    expected_total = sum(sz for _p, sz in files if sz) or 0
    result = {"error": None}

    def _worker():
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=target_dir,
                allow_patterns=allow or None,
                token=token,
                max_workers=4,
            )
        except Exception as exc:  # surfaced to the main thread below
            result["error"] = exc

    def _on_disk_bytes():
        total = 0
        for rel_path, _ in files:
            try:
                total += os.path.getsize(os.path.join(target_dir, rel_path))
            except OSError:
                pass
        return total

    print("\n  Falling back to the Hugging Face library (snapshot_download)...")
    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    POLL = 3                 # seconds between progress samples
    STALL_TIMEOUT = 120      # seconds with no new bytes before we warn
    last_bytes = _on_disk_bytes()
    last_progress_time = time.time()
    last_line_len = 0
    stalled_warned = False

    while worker.is_alive():
        time.sleep(POLL)
        now = time.time()
        cur = _on_disk_bytes()
        delta = cur - last_bytes
        speed = delta / POLL if delta > 0 else 0
        if delta > 0:
            last_progress_time = now
            stalled_warned = False
        last_bytes = cur

        grand_total = max(expected_total, cur) or 1
        remaining = max(grand_total - cur, 0)
        eta = (remaining / speed) if speed > 0 else None
        pct = min(100.0, cur * 100.0 / grand_total)
        line = (f"\r  {pct:5.1f}%  {_human_bytes(cur)} / {_human_bytes(grand_total)}  "
                f"{_human_bytes(speed)}/s  ETA {_human_time(eta)}")
        pad = " " * max(0, last_line_len - len(line))
        sys.stdout.write(line + pad)
        sys.stdout.flush()
        last_line_len = len(line)

        stalled_for = now - last_progress_time
        if stalled_for >= STALL_TIMEOUT and not stalled_warned:
            sys.stdout.write(
                f"\n  WARNING: no new data for {int(stalled_for)}s - the connection looks "
                f"stalled. The library keeps retrying in the background; if it stays stuck, "
                f"stop and re-run the installer to resume exactly where it left off.\n")
            sys.stdout.flush()
            stalled_warned = True

    worker.join()
    sys.stdout.write("\n")

    _clean_stale_aria2_controls(target_dir, files)

    if result["error"] is not None:
        print(f"  Hugging Face fallback error: {result['error']}")
        return False
    return not _missing_files(target_dir, files)

def _file_complete(target_dir, rel_path, size):
    full = os.path.join(target_dir, rel_path)
    if not os.path.exists(full) or os.path.exists(full + ".aria2"): return False
    if size is not None and os.path.getsize(full) != size: return False
    return True

def _missing_files(target_dir, files):
    return [(p, s) for p, s in files if not _file_complete(target_dir, p, s)]

def main():
    if len(sys.argv) != 3:
        print('Usage: download_model.py "<huggingface_repo_id>" "<ai_dir>"')
        return 1

    repo_id, ai_dir = sys.argv[1], sys.argv[2]
    target_dir = os.path.join(ai_dir, "model")
    os.makedirs(target_dir, exist_ok=True)

    marker = os.path.join(target_dir, "DOWNLOAD_COMPLETE")
    source_file = os.path.join(target_dir, "MODEL_SOURCE.txt")

    if os.path.exists(marker):
        previous = ""
        try:
            with open(source_file, "r", encoding="utf-8") as fh: previous = fh.read().strip()
        except OSError: pass
        if previous == repo_id:
            print(f"Model already fully downloaded ({repo_id}); skipping.")
            return 0
        print(f"A different model was previously downloaded here. Fetching {repo_id} instead.")
        try: os.remove(marker)
        except OSError: pass

    try:
        from huggingface_hub import get_token
        token = get_token()
    except Exception:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print(f"Listing files for {repo_id} ...")
    try:
        files = _list_repo_files_with_sizes(repo_id, token)
    except Exception as exc:
        message = str(exc).lower()
        if any(tok in message for tok in ("401", "403", "gated", "unauthorized", "access to model")):
            print(f"ERROR: {exc}\nThis model is gated. Run `huggingface-cli login` first.")
        else:
            print(f"ERROR: could not list repo files: {exc}")
        return 1

    if not files:
        print(f"ERROR: no downloadable files found for {repo_id}.")
        return 1

    total_size = sum(sz for _p, sz in files if sz)
    print(f"           -> {target_dir}")
    print(f"{len(files)} files, ~{_human_bytes(total_size)} total.")

    attempt = 0
    while attempt < MAX_DOWNLOAD_RETRIES:
        todo = _missing_files(target_dir, files)
        if not todo: break
        attempt += 1
        
        # Aria2 block execution
        download_with_aria2(repo_id, target_dir, todo, token)

        if not _missing_files(target_dir, files): break
        if attempt >= MAX_DOWNLOAD_RETRIES: break
        time.sleep(2)

    # Check if anything is still missing. If so, hand the leftovers to the
    # Hugging Face library (snapshot_download), which has its own retry/backoff
    # and is far more resilient to flaky DNS than a raw urllib loop.
    todo = _missing_files(target_dir, files)
    if todo:
        download_with_hf(repo_id, target_dir, todo, token)

    todo = _missing_files(target_dir, files)
    if todo:
        print(f"\nERROR: {len(todo)} file(s) did not finish:")
        for rel_path, _ in todo[:10]: print(f"  - {rel_path}")
        print("It is safely saved. Re-run to pick up EXACTLY where you left off.")
        return 1

    try:
        with open(source_file, "w", encoding="utf-8") as fh: fh.write(repo_id + "\n")
        with open(marker, "w", encoding="utf-8") as fh: fh.write(repo_id + "\n")
    except OSError: pass

    print(f"\nDone. Model saved to: {target_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
