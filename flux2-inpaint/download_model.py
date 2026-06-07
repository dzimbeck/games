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
  and **never deletes** data already downloaded.

Usage::

    python download_model.py "<huggingface_repo_id>" "<ai_dir>"
"""

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

MAX_DOWNLOAD_RETRIES = 4
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
        "--async-dns=false", "--max-file-not-found=5",
        # Anti-hang: the Xet bridge (cas-bridge.xethub.hf.co) frequently accepts a
        # connection and then sends no data, which would otherwise stream 0 B/s
        # forever. --timeout drops a connection that delivers no data for 60s,
        # --connect-timeout bounds the initial handshake, and --lowest-speed-limit
        # tears down and retries a connection that has gone effectively dead
        # (sustained < 1 KiB/s) without harming genuinely slow-but-alive links.
        "--timeout=60", "--connect-timeout=30", "--lowest-speed-limit=1K",
        f"--dir={target_dir}",
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
        warned = set()
        last_line_len = 0
        consecutive_rpc_failures = 0
        # Stall detection: if the total downloaded bytes do not advance at all
        # for STALL_TIMEOUT seconds while files remain, the attempt is wedged
        # (e.g. every connection sitting on a dead Xet-bridge socket). Abort so
        # the outer loop restarts aria2 (resuming from the .aria2 controls) or
        # hands off to the huggingface_hub fallback. This is the backstop behind
        # aria2's own per-connection --timeout/--lowest-speed-limit.
        STALL_TIMEOUT = 150
        max_completed = 0
        last_progress_time = time.monotonic()

        while True:
            if daemon.poll() is not None:
                print("\n  ERROR: aria2 process exited.")
                return False

            try:
                stats = api.get_stats()
            except Exception:
                consecutive_rpc_failures += 1
                if consecutive_rpc_failures >= 20:
                    print("\n  ERROR: lost RPC contact.")
                    return False
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

            now = time.monotonic()
            if completed > max_completed:
                max_completed = completed
                last_progress_time = now
            elif n_done < len(entries) and (now - last_progress_time) > STALL_TIMEOUT:
                print(f"\n  ERROR: download stalled (no progress for "
                      f"{STALL_TIMEOUT}s); restarting to resume.")
                return False

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
                fatal, requeued = [], []
                for entry in errored:
                    rel_path, dl = entry
                    n = file_retries.get(rel_path, 0) + 1
                    file_retries[rel_path] = n
                    
                    # HARD LIMIT: Break the infinite Xet bridge loop
                    if _is_fatal_error(dl.error_message) or n > 12:
                        fatal.append((rel_path, dl.error_message or f"Max transient retries ({n}) exceeded"))
                        continue
                        
                    try: api.remove([dl], force=True, files=False, clean=True)
                    except Exception: pass
                    try:
                        entry[1] = _add_download(rel_path)
                        requeued.append(rel_path)
                    except Exception as exc:
                        if _is_fatal_error(str(exc)): fatal.append((rel_path, str(exc)))
                        else: requeued.append(rel_path)

                if fatal:
                    sys.stdout.write("\n")
                    for rel_path, msg in fatal:
                        print(f"  Aria2 aborted {rel_path}: {msg or 'unknown error'}. Moving to fallback...")
                    return False

                if requeued:
                    # Re-queuing resumes a file from its .aria2 control, so its
                    # reported bytes can briefly dip; give the reconnect room
                    # before the stall detector can fire.
                    last_progress_time = time.monotonic()
                    fresh = [p for p in requeued if p not in warned]
                    warned.update(requeued)
                    if fresh:
                        sys.stdout.write("\n")
                        print(f"  Transient error on {len(fresh)} file(s). Retrying: {', '.join(fresh[:3])}")
                    time.sleep(min(2 + max(file_retries.get(p, 1) for p in requeued), 15))

            if n_done == len(entries):
                sys.stdout.write("\n")
                return True
            time.sleep(0.5)

    except KeyboardInterrupt:
        sys.stdout.write("\n")
        print("Interrupted. Re-run to resume.")
        raise
    finally:
        api_local = locals().get("api")
        if api_local:
            try: api_local.pause_all()
            except Exception: pass
        daemon.terminate()
        try: daemon.wait(timeout=5)
        except Exception: daemon.kill()

def _file_complete(target_dir, rel_path, size):
    full = os.path.join(target_dir, rel_path)
    if not os.path.exists(full):
        return False
    ctrl = full + ".aria2"
    if size is not None:
        # Size is authoritative. A file at the exact expected size is complete
        # even if aria2 left a stale control file behind (this happens when the
        # daemon is terminated the instant a file finishes). Clean the stale
        # control so the file is recognised as done and never re-downloaded.
        if os.path.getsize(full) == size:
            if os.path.exists(ctrl):
                try: os.remove(ctrl)
                except OSError: pass
            return True
        return False
    # Size unknown: only trust files with no leftover .aria2 control file.
    return not os.path.exists(ctrl)

def _missing_files(target_dir, files):
    return [(p, s) for p, s in files if not _file_complete(target_dir, p, s)]

def download_with_hf(repo_id, target_dir, files, token):
    """Finish remaining files with huggingface_hub's snapshot_download.

    This is the authoritative, Xet-aware path. With ``hf_xet`` installed it
    speaks Hugging Face's native Xet protocol (not the flaky cas-bridge HTTP
    proxy aria2 is limited to), and it downloads every file into a private
    ``.cache/huggingface/download/*.incomplete`` temp, only *atomically* moving
    it into place once the content is verified. Consequences that matter here:

    * It **never destroys** data already on disk -- unlike the old hand-rolled
      fallback, there is no ``os.remove`` of a multi-GB partial.
    * It **resumes** its own partial temps across runs and **skips** files that
      are already complete, so finished files are never re-downloaded.
    * It has built-in timeouts and retries, so a dead socket cannot hang it.

    Returns True if every requested file is present afterwards.
    """
    from huggingface_hub import snapshot_download

    patterns = [p for p, _ in files]
    # Remove only stale aria2 *control* files for the handful we are handing off
    # so huggingface_hub manages these cleanly. We deliberately never delete the
    # data files themselves: any complete file is preserved, and hf will resume
    # or atomically replace an incomplete one without losing finished files.
    for rel_path, _ in files:
        ctrl = os.path.join(target_dir, rel_path + ".aria2")
        if os.path.exists(ctrl):
            try: os.remove(ctrl)
            except OSError: pass

    print(f"\n  Finishing {len(files)} file(s) with huggingface_hub "
          f"(native Xet, safe resume)...")
    for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=target_dir,
                allow_patterns=patterns,
                token=token,
                max_workers=4,
            )
        except Exception as exc:  # noqa: BLE001 - classify and retry/report
            if _is_fatal_error(str(exc)):
                print(f"  huggingface_hub: fatal error: {exc}")
                return False
            wait = min(5 * attempt, 30)
            print(f"  huggingface_hub: transient error ({exc}); "
                  f"retry {attempt}/{MAX_DOWNLOAD_RETRIES} in {wait}s...")
            time.sleep(wait)
            continue
        if not _missing_files(target_dir, files):
            return True
    return not _missing_files(target_dir, files)

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
        done_n = len(files) - len(todo)
        if done_n:
            print(f"  {done_n}/{len(files)} file(s) already present; fetching {len(todo)} remaining.")
        
        download_with_aria2(repo_id, target_dir, todo, token)

        if not _missing_files(target_dir, files): break
        if attempt >= MAX_DOWNLOAD_RETRIES: break
        
        wait = min(5 * attempt, 15)
        print(f"  Retrying Aria2 block in {wait}s...")
        time.sleep(wait)

    todo = _missing_files(target_dir, files)
    if todo:
        # aria2 could not finish these (commonly the Xet bridge stalling/expiring
        # signed URLs). Hand them to huggingface_hub, which uses the native Xet
        # protocol, resumes safely and never destroys finished files.
        download_with_hf(repo_id, target_dir, todo, token)

    todo = _missing_files(target_dir, files)
    if todo:
        print(f"\nERROR: {len(todo)} file(s) did not finish:")
        for rel_path, _ in todo[:10]: print(f"  - {rel_path}")
        if len(todo) > 10: print(f"  ... and {len(todo) - 10} more")
        print("It is resumable: re-run install.bat to continue where it left off.")
        return 1

    try:
        with open(source_file, "w", encoding="utf-8") as fh: fh.write(repo_id + "\n")
        with open(marker, "w", encoding="utf-8") as fh: fh.write(repo_id + "\n")
    except OSError: pass

    print(f"\nDone. Model saved to: {target_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
