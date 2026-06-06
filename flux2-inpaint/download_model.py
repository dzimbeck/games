"""Download and save a FLUX.2 model for local image editing / inpainting.

Uses the official FLUX.2 open-weight models from Black Forest Labs:
  https://huggingface.co/black-forest-labs/FLUX.2-klein-4B   (Apache-2.0)
  https://huggingface.co/black-forest-labs/FLUX.2-klein-9B   (non-commercial)
  https://huggingface.co/black-forest-labs/FLUX.2-dev        (non-commercial)

Model overview / which-model-to-use guidance:
  https://github.com/black-forest-labs/flux2

The weights are downloaded in the diffusers layout so they can be loaded with
``Flux2KleinPipeline`` / ``Flux2Pipeline`` from a local folder (no network needed
at inference time).

Robustness
----------
These models are several GB, so a flaky connection used to leave the install in
a broken half-state. This script is now **resumable and idempotent**:

* ``snapshot_download`` only fetches files that are missing or incomplete, so
  re-running picks up exactly where it left off (it reuses the partially
  downloaded ``*.incomplete`` chunks).
* The download is wrapped in a retry loop with exponential backoff so a brief
  network drop is retried automatically instead of aborting the whole install.
* When every file is already present a ``DOWNLOAD_COMPLETE`` marker is written;
  on the next run we detect it and skip the network entirely.
"""

import os
import sys
import time

from huggingface_hub import snapshot_download

# Patterns we intentionally do not download (single-file checkpoints used by
# other runtimes such as ComfyUI / the reference CLI). We only need the
# diffusers layout.
IGNORE_PATTERNS = ["*.gguf", "flux2-*.safetensors", "flux-2-*.safetensors"]

MAX_ATTEMPTS = 8
INITIAL_BACKOFF = 5  # seconds; doubles each retry, capped at MAX_BACKOFF
MAX_BACKOFF = 120


def _enable_hf_transfer_if_available():
    """Use hf_transfer (faster, more resilient) when it is installed."""
    try:
        import hf_transfer  # noqa: F401
    except Exception:  # noqa: BLE001 - it is optional
        return
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def download_with_resume(repo_id, target_dir):
    """Download ``repo_id`` into ``target_dir``, resuming + retrying as needed.

    Returns True on success, False if every attempt failed.
    """
    _enable_hf_transfer_if_available()

    attempt = 0
    backoff = INITIAL_BACKOFF
    while attempt < MAX_ATTEMPTS:
        attempt += 1
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=target_dir,
                ignore_patterns=IGNORE_PATTERNS,
                # snapshot_download resumes partial files by default; this keeps
                # the behaviour explicit and tolerant of slow CDNs.
                max_workers=4,
                etag_timeout=30,
            )
            return True
        except KeyboardInterrupt:
            print("\nInterrupted. Re-run install.bat to resume the download.")
            raise
        except Exception as exc:  # noqa: BLE001 - retry transient failures
            print(f"  Attempt {attempt}/{MAX_ATTEMPTS} failed: "
                  f"{type(exc).__name__}: {exc}")
            # Auth / "file not found" style errors will not be fixed by retrying.
            message = str(exc).lower()
            if any(tok in message for tok in
                   ("401", "403", "gated", "unauthorized", "access to model",
                    "awaiting a review", "not found", "404")):
                print("  This looks like an access/permission problem, not a "
                      "network blip; not retrying.")
                return False
            if attempt >= MAX_ATTEMPTS:
                break
            print(f"  Retrying in {backoff}s (resuming where it left off)...")
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)
    return False


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

    print(f"Downloading {repo_id}")
    print(f"           -> {target_dir}")
    print("This can take a while (these models are several GB).")
    print("If your connection drops, just run install.bat again -- it resumes.")

    if not download_with_resume(repo_id, target_dir):
        print()
        print(f"ERROR: Failed to download {repo_id} after {MAX_ATTEMPTS} attempts.")
        print()
        print("Common causes:")
        print("  - Gated/non-commercial model: accept its license on the model's")
        print("    Hugging Face page, then authenticate with `huggingface-cli login`.")
        print("  - Network/proxy issues or insufficient disk space.")
        print("  - The download is resumable: re-run install.bat to continue.")
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
