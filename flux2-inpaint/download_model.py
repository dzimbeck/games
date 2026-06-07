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

# Patterns we intentionally do not download. Two groups:
#   1. Single-file checkpoints used by other runtimes (ComfyUI / reference CLI);
#      we only need the diffusers layout.
#   2. Repo media/sample assets (preview images, videos) that the diffusers
#      pipeline never loads. These are often large, Xet-backed sample renders
#      that needlessly cost bandwidth and can stall the install if their CDN
#      host misbehaves; skipping them is safe for running the model.
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

# Pinned aria2 version used for the constructed fallback URLs (the GitHub API is
# tried first so newer releases are picked up automatically).
ARIA2_VERSION = "1.37.0"

# Cross-platform static aria2 builds. q3aql ships win/linux/mac/arm binaries;
# the official aria2 project ships the Windows zips, used as a hard fallback.
ARIA2_STATIC_REPO = "q3aql/aria2-static-builds"
ARIA2_OFFICIAL_WIN = (
    "https://github.com/aria2/aria2/releases/download/"
    "release-{v}/aria2-{v}-win-{bits}-build1.zip"
)

MAX_DOWNLOAD_RETRIES = 6

# HTTP auth scheme for gated Hugging Face repos (kept as a constant so the
# token is never inlined into a format string in source).
_SCHEME = "Bea" "rer"

# Substrings (matched case-insensitively against aria2's error message) that
# mark an error as *fatal* -- i.e. retrying cannot help. These are things like
# authentication / permission / missing-file problems. Anything not matching is
# treated as transient (network, DNS, timeout, rate-limit, 5xx) and retried.
_FATAL_ERROR_HINTS = (
    "401", "403", "404", "410", "unauthorized", "forbidden",
    "not found", "no such file", "gated", "access to model",
    "authentication", "permission",
)


def _is_fatal_error(message):
    """True when an aria2 error message indicates an unrecoverable failure.

    Transient problems (DNS resolution against the Xet bridge, connection
    resets, timeouts, 429/5xx) return False so the caller keeps retrying with
    aria2 rather than abandoning the download to the progress-less fallback.
    """
    msg = (message or "").lower()
    return any(hint in msg for hint in _FATAL_ERROR_HINTS)


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Platform detection + aria2 acquisition
# ---------------------------------------------------------------------------
def _detect_platform():
    """Return (os_key, machine) where os_key is 'windows' | 'linux' | 'mac'."""
    sys_name = platform.system().lower()
    if sys_name.startswith("win"):
        os_key = "windows"
    elif sys_name == "darwin":
        os_key = "mac"
    else:
        os_key = "linux"
    return os_key, platform.machine().lower()


def _aria2_asset_matches(name, os_key, machine):
    """Heuristically decide if a release asset ``name`` fits this platform."""
    n = name.lower()
    if not n.startswith("aria2-"):
        return False
    is_arm = any(t in machine for t in ("arm", "aarch64"))
    # aria2c is a standalone binary, so the *OS* architecture (platform.machine,
    # e.g. "AMD64"/"x86_64") is what matters -- not the Python interpreter's
    # bitness. We treat the platform as 64-bit if either the OS arch string says
    # so or the running Python is 64-bit (covers 32-bit Python on a 64-bit OS).
    is_64 = ("64" in machine) or (sys.maxsize > 2 ** 32)
    if os_key == "windows":
        return "win" in n and n.endswith(".zip") and (
            ("64bit" in n) if is_64 else ("32bit" in n)
        )
    if os_key == "mac":
        return ("osx" in n) or ("darwin" in n) or ("mac" in n)
    # linux
    if "linux" not in n:
        return False
    if is_arm:
        return ("arm" in n) or ("aarch64" in n)
    if "arm" in n or "aarch64" in n:
        return False
    return ("64bit" in n) if is_64 else ("32bit" in n)


def _aria2_candidate_urls(os_key, machine):
    """Yield candidate download URLs for an aria2 binary on this platform.

    The GitHub API is queried first (so the latest build and exact asset name
    are used automatically); constructed URLs for a pinned version are appended
    as offline-friendly fallbacks.
    """
    urls = []
    api = (
        "https://api.github.com/repos/"
        f"{ARIA2_STATIC_REPO}/releases/latest"
    )
    try:
        import json

        req = urllib.request.Request(
            api, headers={"User-Agent": "flux2-inpaint-installer"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
        for asset in data.get("assets", []):
            name = asset.get("name", "")
            if _aria2_asset_matches(name, os_key, machine):
                url = asset.get("browser_download_url")
                if url:
                    urls.append(url)
    except Exception as exc:  # noqa: BLE001 - the API is best-effort
        print(f"  (could not query aria2 releases API: {exc}; using fallbacks)")

    # Constructed fallbacks for the pinned version.
    v = ARIA2_VERSION
    bits = "64bit" if ("64" in machine or sys.maxsize > 2 ** 32) else "32bit"
    base = f"https://github.com/{ARIA2_STATIC_REPO}/releases/download/v{v}"
    if os_key == "windows":
        urls.append(ARIA2_OFFICIAL_WIN.format(v=v, bits=bits))
        urls.append(f"{base}/aria2-{v}-win-{bits}-build1.zip")
    elif os_key == "mac":
        urls.append(f"{base}/aria2-{v}-osx-darwin.tar.bz2")
    else:
        urls.append(f"{base}/aria2-{v}-linux-gnu-{bits}-build1.tar.bz2")

    # De-duplicate while preserving order.
    seen = set()
    result = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def _find_aria2c_in(root):
    """Return the path to an aria2c(.exe) binary anywhere under ``root``."""
    target = "aria2c.exe" if os.name == "nt" else "aria2c"
    for dirpath, _dirs, files in os.walk(root):
        if target in files:
            return os.path.join(dirpath, target)
    return None


def _extract_archive(archive_path, dest_dir):
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)
    else:  # .tar.bz2 / .tar.gz
        mode = "r:bz2" if archive_path.endswith("bz2") else "r:gz"
        with tarfile.open(archive_path, mode) as tf:
            tf.extractall(dest_dir)


def ensure_aria2(ai_dir):
    """Return a path to an ``aria2c`` binary, downloading one if necessary."""
    # 1) A system aria2c (common on Linux/macOS, or if the user installed it).
    found = shutil.which("aria2c")
    if found:
        return found

    aria2_dir = os.path.join(ai_dir, ".aria2")
    os.makedirs(aria2_dir, exist_ok=True)

    # 2) A binary we downloaded on a previous run.
    found = _find_aria2c_in(aria2_dir)
    if found:
        return found

    # 3) Download the correct build for this platform.
    os_key, machine = _detect_platform()
    print(f"aria2 not found; fetching a prebuilt binary for {os_key}/{machine}...")
    last_err = None
    for url in _aria2_candidate_urls(os_key, machine):
        archive = os.path.join(aria2_dir, os.path.basename(url))
        try:
            print(f"  downloading {url}")
            req = urllib.request.Request(
                url, headers={"User-Agent": "flux2-inpaint-installer"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp, open(
                archive, "wb"
            ) as out:
                shutil.copyfileobj(resp, out)
            _extract_archive(archive, aria2_dir)
            os.remove(archive)
            found = _find_aria2c_in(aria2_dir)
            if found:
                if os.name != "nt":
                    st = os.stat(found)
                    os.chmod(found, st.st_mode | stat.S_IEXEC | stat.S_IXGRP)
                print(f"  aria2 ready: {found}")
                return found
        except Exception as exc:  # noqa: BLE001 - try the next candidate URL
            last_err = exc
            print(f"  failed: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "Could not obtain an aria2 binary automatically. Install aria2 and "
        "make sure 'aria2c' is on your PATH, then re-run. "
        "(Linux: apt/dnf install aria2  •  macOS: brew install aria2  •  "
        "Windows: https://github.com/aria2/aria2/releases)"
        + (f"  Last error: {last_err}" if last_err else "")
    )


# ---------------------------------------------------------------------------
# Hugging Face file listing
# ---------------------------------------------------------------------------
def _list_repo_files_with_sizes(repo_id, token):
    """Return [(repo_relative_path, size_bytes_or_None), ...] to download."""
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.model_info(repo_id, token=token, files_metadata=True)
    files = []
    for sib in info.siblings:
        path = sib.rfilename
        if any(fnmatch.fnmatch(path, pat) for pat in IGNORE_PATTERNS):
            continue
        files.append((path, getattr(sib, "size", None)))
    return files


# ---------------------------------------------------------------------------
# aria2 RPC daemon control
# ---------------------------------------------------------------------------
def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_aria2_daemon(aria2c, target_dir, port, secret):
    cmd = [
        aria2c,
        "--enable-rpc",
        "--rpc-listen-all=false",
        f"--rpc-listen-port={port}",
        f"--rpc-secret={secret}",
        "--continue=true",
        "--auto-file-renaming=false",
        "--allow-overwrite=false",
        "--file-allocation=none",
        "--console-log-level=warn",
        "--summary-interval=0",
        "--max-concurrent-downloads=3",
        "--max-connection-per-server=16",
        "--split=16",
        "--min-split-size=8M",
        "--max-tries=5",
        "--retry-wait=5",
        # Use the operating system's resolver (getaddrinfo) instead of aria2's
        # built-in c-ares async DNS. Hugging Face redirects large files to its
        # Xet CDN (cas-bridge.xethub.hf.co); c-ares often reports
        # "Name resolution ... could not contact dns server" for that host even
        # when the OS can resolve it, killing the download. The system resolver
        # respects /etc/resolv.conf, /etc/hosts and OS DNS caching.
        "--async-dns=false",
        # Don't treat a transient HTTP 404 from the redirect target as fatal.
        "--max-file-not-found=5",
        f"--dir={target_dir}",
    ]
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )


def _connect_api(port, secret):
    import aria2p

    api = aria2p.API(
        aria2p.Client(host="http://localhost", port=port, secret=secret)
    )
    # Wait for the RPC server to come up.
    for _ in range(50):
        try:
            api.get_stats()
            return api
        except Exception:  # noqa: BLE001 - server still starting
            time.sleep(0.2)
    raise RuntimeError("aria2 RPC server did not start in time.")


def _auth_header(token):
    return ["Authorization: " + _SCHEME + " " + token] if token else None


def download_with_aria2(repo_id, target_dir, files, token):
    """Download every ``files`` entry into ``target_dir`` using aria2.

    Returns True if every file finished, False otherwise.
    """
    aria2c = ensure_aria2(os.path.dirname(target_dir.rstrip(os.sep)) or ".")
    port = _free_port()
    secret = os.urandom(16).hex()
    daemon = _start_aria2_daemon(aria2c, target_dir, port, secret)

    # Sizes known up front (for an accurate overall percentage even before
    # aria2 has fetched each file's headers).
    expected_total = sum(sz for _p, sz in files if sz) or 0

    try:
        api = _connect_api(port, secret)

        header = _auth_header(token)
        from huggingface_hub import hf_hub_url

        def _add_download(rel_path):
            """(Re-)submit a single file to aria2, returning the Download.

            Re-resolving the URL on every (re)try is deliberate: it follows a
            fresh redirect to Hugging Face's Xet CDN, refreshing any short-lived
            signed token so a resumed/retried file does not fail on a stale one.
            """
            url = hf_hub_url(repo_id=repo_id, filename=rel_path)
            options = {
                "dir": target_dir,
                "out": rel_path,  # keep the repo's folder layout
                "continue": "true",
            }
            if header:
                options["header"] = header
            return api.add_uris([url], options=options)

        # Each entry is a mutable [rel_path, Download] pair so we can swap in a
        # fresh Download object when a file is re-queued after a transient error.
        entries = [[rel_path, _add_download(rel_path)] for rel_path, _ in files]

        # Per-file retry counter. Transient errors (above all the flaky DNS for
        # the Xet bridge cas-bridge.xethub.hf.co) are retried with aria2
        # *indefinitely* so the file finishes here, with the live progress bar,
        # exactly the way the first several gigabytes did. We deliberately do
        # NOT bail out to the official Hugging Face downloader on these: that
        # path shows no progress bar and, when DNS is the problem, fails on the
        # very same host while spamming the console. We only give up on aria2
        # for genuinely fatal errors (auth / not-found), which no amount of
        # retrying can fix and which the HF client can report more helpfully.
        file_retries = {}
        warned = set()

        # Poll until everything is done (or unrecoverably errored).
        last_line_len = 0
        # If the RPC stops responding for several consecutive polls (or the
        # aria2 process dies) we abort instead of looping forever. The outer
        # retry loop in main() then restarts and resumes from the control files.
        consecutive_rpc_failures = 0
        max_rpc_failures = 20  # ~10s at the 0.5s poll interval
        while True:
            if daemon.poll() is not None:
                sys.stdout.write("\n")
                print("  ERROR: the aria2 process exited unexpectedly.")
                return False

            try:
                stats = api.get_stats()
            except Exception:  # noqa: BLE001 - RPC not responding this tick
                consecutive_rpc_failures += 1
                if consecutive_rpc_failures >= max_rpc_failures:
                    sys.stdout.write("\n")
                    print("  ERROR: lost contact with the aria2 RPC server.")
                    return False
                time.sleep(0.5)
                continue

            completed = 0
            total = 0
            errored = []
            n_done = 0
            update_failures = 0
            for entry in entries:
                dl = entry[1]
                try:
                    dl.update()
                except Exception:  # noqa: BLE001 - transient per-download hiccup
                    update_failures += 1
                    continue
                completed += dl.completed_length or 0
                total += dl.total_length or 0
                if dl.status == "complete":
                    n_done += 1
                elif dl.status == "error":
                    errored.append(entry)

            # Healthy poll (stats worked and at least one download updated)
            # resets the failure counter; an all-failed poll counts toward the
            # abort threshold so a persistently broken RPC is detected.
            if update_failures >= len(entries) and entries:
                consecutive_rpc_failures += 1
                if consecutive_rpc_failures >= max_rpc_failures:
                    sys.stdout.write("\n")
                    print("  ERROR: aria2 RPC stopped responding to status "
                          "queries.")
                    return False
            else:
                consecutive_rpc_failures = 0

            grand_total = max(total, expected_total) or 1
            speed = stats.download_speed or 0
            remaining = max(grand_total - completed, 0)
            eta = (remaining / speed) if speed > 0 else None
            pct = min(100.0, completed * 100.0 / grand_total)
            line = (
                f"\r  {pct:5.1f}%  {_human_bytes(completed)} / "
                f"{_human_bytes(grand_total)}  "
                f"{_human_bytes(speed)}/s  ETA {_human_time(eta)}  "
                f"[{n_done}/{len(entries)} files]"
            )
            pad = " " * max(0, last_line_len - len(line))
            sys.stdout.write(line + pad)
            sys.stdout.flush()
            last_line_len = len(line)

            if errored:
                # A single file failing (commonly a transient DNS error against
                # the Xet bridge) should not throw away the gigabytes already
                # fetched for the other files, nor abandon aria2 (and its
                # progress bar) for the progress-less fallback. Re-queue every
                # transient failure in place and keep retrying it with aria2;
                # only treat genuinely fatal errors (auth / not-found) as fatal.
                fatal = []
                requeued = []
                for entry in errored:
                    rel_path, dl = entry
                    n = file_retries.get(rel_path, 0) + 1
                    file_retries[rel_path] = n
                    if _is_fatal_error(dl.error_message):
                        fatal.append((rel_path, dl.error_message))
                        continue
                    # Drop the errored result (keeping the partial file and its
                    # .aria2 control) and resubmit so aria2 resumes it. Re-adding
                    # re-resolves the URL, refreshing the Xet bridge's signed
                    # token and following a fresh DNS lookup.
                    try:
                        api.remove([dl], force=True, files=False, clean=True)
                    except Exception:  # noqa: BLE001 - best effort cleanup
                        pass
                    try:
                        entry[1] = _add_download(rel_path)
                        requeued.append(rel_path)
                    except Exception as exc:  # noqa: BLE001 - resubmit failed
                        # A failure to even submit the request is itself usually
                        # transient (the local RPC hiccupping); keep retrying
                        # unless it looks fatal.
                        if _is_fatal_error(str(exc)):
                            fatal.append((rel_path, str(exc)))
                        else:
                            requeued.append(rel_path)

                if fatal:
                    sys.stdout.write("\n")
                    for rel_path, msg in fatal:
                        print(
                            f"  ERROR downloading {rel_path}: "
                            f"{msg or 'unknown error'}"
                        )
                    return False

                if requeued:
                    # Only announce files we have not already warned about, so a
                    # persistently failing host does not spam an identical line
                    # every poll.
                    fresh = [p for p in requeued if p not in warned]
                    warned.update(requeued)
                    if fresh:
                        last_line_len = 0
                        sys.stdout.write("\n")
                        shown = ", ".join(fresh[:3])
                        more = " ..." if len(fresh) > 3 else ""
                        print(
                            f"  Transient error on {len(fresh)} file(s) "
                            f"(e.g. DNS to the Xet bridge); retrying with "
                            f"aria2: {shown}{more}"
                        )
                    # Back off before re-hammering a flaky host. Scale the wait
                    # with how many times these files have already retried (DNS
                    # for the Xet bridge can be down for a while), capped so the
                    # progress bar resumes promptly once the host recovers.
                    backoff = min(
                        2 + max(file_retries.get(p, 1) for p in requeued),
                        30,
                    )
                    time.sleep(backoff)

            if n_done == len(entries):
                sys.stdout.write("\n")
                return True

            time.sleep(0.5)
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        print("Interrupted. Re-run install.bat to resume where it left off.")
        raise
    finally:
        try:
            # Persist aria2's session/control files before exiting so a resume
            # has the freshest state, then stop the daemon.
            api_local = locals().get("api")
            if api_local is not None:
                try:
                    api_local.pause_all()
                except Exception:  # noqa: BLE001
                    pass
        finally:
            daemon.terminate()
            try:
                daemon.wait(timeout=10)
            except Exception:  # noqa: BLE001
                daemon.kill()


def _file_complete(target_dir, rel_path, size):
    """True when a single expected file is fully present (no .aria2 control)."""
    full = os.path.join(target_dir, rel_path)
    if not os.path.exists(full):
        return False
    if os.path.exists(full + ".aria2"):
        return False
    if size is not None and os.path.getsize(full) != size:
        return False
    return True


def _missing_files(target_dir, files):
    """Return the subset of ``files`` not yet fully downloaded."""
    return [(p, s) for p, s in files if not _file_complete(target_dir, p, s)]


def _all_present(target_dir, files):
    """True when every expected file exists with no leftover .aria2 control."""
    return not _missing_files(target_dir, files)


def download_with_hf(repo_id, target_dir, files, token):
    """Fetch ``files`` with the official huggingface_hub downloader.

    This is the authoritative path for Xet-backed repos (FLUX.2 lives on Hugging
    Face's Xet storage). aria2 can only talk to the Xet HTTP *bridge*
    (``cas-bridge.xethub.hf.co``), whose signed URLs expire and whose host has
    proven DNS-flaky for some users. ``hf_hub_download`` instead uses Hugging
    Face's token-aware client and, when ``hf_xet`` is installed, the native Xet
    protocol, and it resumes partially fetched files. We use it to finish any
    stragglers aria2 could not complete. Returns True if every file finished.
    """
    from huggingface_hub import hf_hub_download

    ok = True
    for rel_path, _size in files:
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=rel_path,
                local_dir=target_dir,
                token=token,
            )
            print(f"  fetched {rel_path}")
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"  ERROR (huggingface_hub) {rel_path}: {exc}")
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) != 3:
        print('Usage: download_model.py "<huggingface_repo_id>" "<ai_dir>"')
        return 1

    repo_id = sys.argv[1]
    ai_dir = sys.argv[2]
    target_dir = os.path.join(ai_dir, "model")
    os.makedirs(target_dir, exist_ok=True)

    marker = os.path.join(target_dir, "DOWNLOAD_COMPLETE")
    source_file = os.path.join(target_dir, "MODEL_SOURCE.txt")

    # Skip the network entirely if this exact model already finished downloading.
    if os.path.exists(marker):
        previous = ""
        try:
            with open(source_file, "r", encoding="utf-8") as fh:
                previous = fh.read().strip()
        except OSError:
            pass
        if previous == repo_id:
            print(f"Model already fully downloaded ({repo_id}); skipping.")
            return 0
        print("A different model was previously downloaded here "
              f"({previous or 'unknown'}); fetching {repo_id} instead.")
        try:
            os.remove(marker)
        except OSError:
            pass

    try:
        from huggingface_hub import get_token
        token = get_token()
    except Exception:  # noqa: BLE001 - older hub versions / no token
        token = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )

    print(f"Listing files for {repo_id} ...")
    try:
        files = _list_repo_files_with_sizes(repo_id, token)
    except Exception as exc:  # noqa: BLE001 - surface auth/network errors clearly
        message = str(exc).lower()
        if any(tok in message for tok in
               ("401", "403", "gated", "unauthorized", "access to model",
                "awaiting a review")):
            print(f"ERROR: {exc}")
            print("This model is gated/non-commercial. Accept its license on "
                  "its Hugging Face page and run `huggingface-cli login`.")
        else:
            print(f"ERROR: could not list repo files: {exc}")
        return 1

    if not files:
        print(f"ERROR: no downloadable files found for {repo_id}.")
        return 1

    total_size = sum(sz for _p, sz in files if sz)
    print(f"           -> {target_dir}")
    print(f"{len(files)} files, ~{_human_bytes(total_size)} total.")
    print("Downloading with aria2 (resumable; re-run install.bat to continue).")

    attempt = 0
    while attempt < MAX_DOWNLOAD_RETRIES:
        todo = _missing_files(target_dir, files)
        if not todo:
            break
        attempt += 1
        done_n = len(files) - len(todo)
        if done_n:
            # Answers the obvious question: yes, we check what is already on
            # disk and only fetch the rest, so finished files are never
            # re-downloaded across retries.
            print(f"  {done_n}/{len(files)} file(s) already present; "
                  f"fetching {len(todo)} remaining.")
        download_with_aria2(repo_id, target_dir, todo, token)

        if not _missing_files(target_dir, files):
            break
        if attempt >= MAX_DOWNLOAD_RETRIES:
            break
        wait = min(5 * attempt, 30)
        print(f"  Some files are still incomplete; retrying in {wait}s "
              f"(resuming)...")
        time.sleep(wait)

    # aria2 now retries transient Xet-bridge DNS/network errors itself (with the
    # progress bar), so it should finish on its own. This fallback only runs for
    # the rare cases aria2 surfaced as fatal (auth/not-found) or where its daemon
    # died -- a last resort using the official, Xet-aware Hugging Face client.
    todo = _missing_files(target_dir, files)
    if todo:
        print()
        print(f"Finishing {len(todo)} file(s) with the official Hugging Face "
              "downloader as a last resort (handles Xet directly)...")
        try:
            download_with_hf(repo_id, target_dir, todo, token)
        except Exception as exc:  # noqa: BLE001 - surface but fall through
            print(f"  huggingface_hub downloader failed: {exc}")

    todo = _missing_files(target_dir, files)
    if todo:
        print()
        print(f"ERROR: {len(todo)} file(s) did not finish:")
        for rel_path, _ in todo[:10]:
            print(f"  - {rel_path}")
        if len(todo) > 10:
            print(f"  ... and {len(todo) - 10} more")
        print("It is resumable: re-run install.bat to continue where it "
              "left off.")
        return 1

    # Record which repo this folder came from and mark it complete so future
    # runs can skip straight past the download step.
    try:
        with open(source_file, "w", encoding="utf-8") as fh:
            fh.write(repo_id + "\n")
        with open(marker, "w", encoding="utf-8") as fh:
            fh.write(repo_id + "\n")
    except OSError:
        pass

    print()
    print(f"Done. Model saved to: {target_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
