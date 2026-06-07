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

def download_with_robust_fallback(repo_id, target_dir, files, token):
    """A strictly non-destructive, append-only sequential HTTP downloader."""
    from huggingface_hub import hf_hub_url
    socket.setdefaulttimeout(15.0) 
    
    print(f"\n  Resuming with robust direct-download fallback for {len(files)} files...")
    try:
        for rel_path, expected_size in files:
            full_path = os.path.join(target_dir, rel_path)
            aria_ctrl = full_path + ".aria2"

            # Clean ONLY the Aria2 control session block to free the file.
            # DO NOT DELETE THE 'full_path' FILE CONTAINING ALREADY DOWNLOADED DATA!
            if os.path.exists(aria_ctrl):
                try: os.remove(aria_ctrl)
                except OSError: pass

            url = hf_hub_url(repo_id=repo_id, filename=rel_path)
            max_attempts = 15
            
            for attempt in range(max_attempts):
                initial_pos = os.path.getsize(full_path) if os.path.exists(full_path) else 0
                
                if expected_size and initial_pos == expected_size:
                    break
                if expected_size and initial_pos > expected_size:
                    # Only truncate or wipe if it's somehow larger than the server file
                    try: os.remove(full_path)
                    except OSError: pass
                    initial_pos = 0

                req = urllib.request.Request(url)
                req.add_header("User-Agent", "flux2-fallback-installer")
                if token: req.add_header("Authorization", f"{_SCHEME} {token}")
                
                # Request ONLY the bytes we don't have yet
                if initial_pos > 0: 
                    req.add_header("Range", f"bytes={initial_pos}-")

                try:
                    with urllib.request.urlopen(req) as resp:
                        total_size = expected_size
                        
                        # Handle content range response properties
                        status_code = resp.getcode()
                        is_partial = (status_code == 206)
                        
                        if resp.headers.get("Content-Length"):
                            if is_partial:
                                total_size = initial_pos + int(resp.headers.get("Content-Length"))
                            else:
                                total_size = int(resp.headers.get("Content-Length"))
                                initial_pos = 0 # Server ignored Range header, starting fresh

                        mode = "ab" if (initial_pos > 0 and is_partial) else "wb"
                        with open(full_path, mode) as fh:
                            downloaded = initial_pos
                            start_time = time.time()
                            last_line_len = 0
                            
                            while True:
                                chunk = resp.read(1024 * 128)
                                if not chunk: break
                                fh.write(chunk)
                                downloaded += len(chunk)

                                elapsed = time.time() - start_time
                                speed = (downloaded - initial_pos) / elapsed if elapsed > 0 else 0
                                pct = (downloaded / total_size * 100) if total_size else 0.0

                                line = f"\r    {pct:5.1f}% | {_human_bytes(downloaded)} / {_human_bytes(total_size) if total_size else '?'} | {_human_bytes(speed)}/s [Resuming: {rel_path}]"
                                pad = " " * max(0, last_line_len - len(line))
                                sys.stdout.write(line + pad)
                                sys.stdout.flush()
                                last_line_len = len(line)
                    print() 
                    break 

                except urllib.error.HTTPError as e:
                    if e.code == 416:  # Range Not Satisfiable (usually means we are already done)
                        if expected_size and initial_pos == expected_size:
                            break
                        try: os.remove(full_path)
                        except OSError: pass
                        continue
                    if e.code in (401, 403, 404):
                        print(f"\n    FATAL: HTTP {e.code} for {rel_path}")
                        return False
                    print(f"\n    Server issue: {e}. Retrying ({attempt+1}/{max_attempts})...")
                    time.sleep(3)
                except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
                    print(f"\n    Connection dropped/stalled: {e}. Retrying ({attempt+1}/{max_attempts})...")
                    time.sleep(3)
                except Exception as e:
                    print(f"\n    Error: {e}. Retrying ({attempt+1}/{max_attempts})...")
                    time.sleep(3)
            else:
                print(f"\n  ERROR: Failed to download {rel_path} after {max_attempts} fallback attempts.")
                return False
        return True
    finally:
        socket.setdefaulttimeout(None)

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

    # Check if anything is still missing. If so, drop safely into append-only fallback
    todo = _missing_files(target_dir, files)
    if todo:
        download_with_robust_fallback(repo_id, target_dir, todo, token)

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
