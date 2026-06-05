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
"""

import os
import sys

from huggingface_hub import snapshot_download


def main():
    if len(sys.argv) != 3:
        print('Usage: download_model.py "<huggingface_repo_id>" "<ai_dir>"')
        return 1

    repo_id = sys.argv[1]
    ai_dir = sys.argv[2]
    target_dir = os.path.join(ai_dir, "model")
    os.makedirs(target_dir, exist_ok=True)

    print(f"Downloading {repo_id}")
    print(f"           -> {target_dir}")
    print("This can take a while (these models are several GB)...")

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            # Skip the raw single-file checkpoints used by other runtimes
            # (ComfyUI, the reference CLI); we only need the diffusers layout.
            ignore_patterns=["*.gguf", "flux2-*.safetensors", "flux-2-*.safetensors"],
        )
    except Exception as exc:  # noqa: BLE001 - surface a friendly message
        print(f"ERROR: Failed to download {repo_id}: {type(exc).__name__}: {exc}")
        print()
        print("Common causes:")
        print("  - Gated/non-commercial model: accept its license on the model's")
        print("    Hugging Face page, then authenticate with `huggingface-cli login`.")
        print("  - Network/proxy issues or insufficient disk space.")
        return 1

    # Record which repo this folder came from for reference.
    try:
        with open(os.path.join(target_dir, "MODEL_SOURCE.txt"), "w", encoding="utf-8") as fh:
            fh.write(repo_id + "\n")
    except OSError:
        pass

    print()
    print(f"Done. Model saved to: {target_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
