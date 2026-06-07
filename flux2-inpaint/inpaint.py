"""Local image editing / inpainting with FLUX.2.

This is the inference half of the one-click installer. ``install.bat`` downloads
a FLUX.2 model into ``ai-model/model`` and writes ``run_inpaint.bat`` with the
right ``--pipeline`` / ``--mode`` / ``--steps`` for the VRAM you reported.

Modes of operation
------------------
* Text-to-image:   pass ``--prompt`` only.
* Instruction edit: pass ``--image`` (one or more reference images) + ``--prompt``.
* Masked inpaint:   additionally pass ``--mask`` (white = the region to change).
                    The edited result is composited back onto the original so
                    only the masked area changes.

FLUX.2 reference:
  https://github.com/black-forest-labs/flux2
  https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
"""

import argparse
import os
import platform
import sys


def _default_cuda_alloc_conf():
    """Pick a CUDA allocator config that actually works on this OS.

    ``expandable_segments:True`` is the best anti-fragmentation option, but it is
    **Linux-only** -- on Windows PyTorch prints "expandable_segments not supported
    on this platform" and ignores it. So we only request it on Linux and fall back
    to ``max_split_size_mb`` + ``garbage_collection_threshold`` (both cross-platform)
    elsewhere, which cap fragmentation without the unsupported feature.
    """
    if platform.system() == "Linux":
        return "expandable_segments:True"
    # Windows / macOS: limit oversized split blocks and let the allocator reclaim
    # cached memory early once it gets ~80% full. Helps the long-lived GUI that
    # runs many generations from one loaded pipeline.
    return "max_split_size_mb:128,garbage_collection_threshold:0.8"


# Must be set before torch initialises CUDA, so it happens at import time and only
# fills in a default the user can still override from the environment.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", _default_cuda_alloc_conf())

import torch
from PIL import Image


def round_to(value, multiple=16):
    return max(multiple, int(round(value / multiple) * multiple))


def load_pipeline(model_dir, pipeline_kind, mode):
    """Build the FLUX.2 diffusers pipeline for the requested model + mode."""
    if pipeline_kind == "dev":
        from diffusers import Flux2Pipeline as PipelineCls
    else:
        from diffusers import Flux2KleinPipeline as PipelineCls

    torch_dtype = torch.bfloat16
    quantization_config = None

    if mode == "quant":
        # 4-bit (bitsandbytes) quantization of the heavy components so the model
        # fits on small GPUs. Falls back to plain CPU offload if unavailable.
        try:
            from diffusers import PipelineQuantizationConfig

            quantization_config = PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                },
                components_to_quantize=["transformer", "text_encoder"],
            )
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: 4-bit quantization unavailable ({exc}).")
            print("         Install it with `pip install bitsandbytes` for lower VRAM use.")
            print("         Falling back to CPU offload for now.")
            quantization_config = None
            mode = "offload"

    kwargs = {"torch_dtype": torch_dtype}
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config

    pipe = PipelineCls.from_pretrained(model_dir, **kwargs)

    if mode == "cuda":
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available; falling back to CPU offload.")
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
    else:
        # offload + quant both stream components between CPU and GPU as needed.
        pipe.enable_model_cpu_offload()

    # Decode the latents in tiles/slices instead of all at once. The VAE decode
    # is a major source of the end-of-run memory spike, so this keeps generation
    # within budget on 12 GB cards (and is harmless on larger ones). Guarded with
    # hasattr because not every diffusers build exposes both helpers.
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    return pipe


def generate_image(
    pipe,
    prompt,
    ref_images=None,
    mask=None,
    width=None,
    height=None,
    steps=4,
    guidance=4.0,
    seed=42,
):
    """Run one FLUX.2 generation and return a ``PIL.Image``.

    This is the reusable core shared by the command-line ``main`` and by the
    GUI / any future API. It mirrors the three CLI modes:

    * ``ref_images is None``            -> text-to-image.
    * ``ref_images`` given             -> instruction edit using the references.
    * ``ref_images`` + ``mask`` given  -> masked inpaint: the edit is composited
      back onto the first reference image so only the white area of ``mask``
      changes.

    ``ref_images`` and ``mask`` may be file paths or ``PIL.Image`` objects.
    """
    base_image = None
    loaded_refs = None
    if ref_images is not None:
        loaded_refs = [
            img if isinstance(img, Image.Image) else Image.open(img)
            for img in ref_images
        ]
        loaded_refs = [img.convert("RGB") for img in loaded_refs]
        base_image = loaded_refs[0]
        if width is None or height is None:
            width, height = round_to(base_image.width), round_to(base_image.height)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(seed)

    call_kwargs = {
        "prompt": prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "generator": generator,
    }
    if loaded_refs is not None:
        call_kwargs["image"] = loaded_refs
    if width is not None and height is not None:
        call_kwargs["width"] = width
        call_kwargs["height"] = height

    # The GUI keeps one pipeline loaded and runs many generations, so release any
    # cached-but-unused blocks from the previous run before this one. This hands
    # fragmented reserved memory back to the allocator and is the cheapest way to
    # avoid a slow creep toward OOM over a long painting session.
    if device == "cuda":
        torch.cuda.empty_cache()

    try:
        result = pipe(**call_kwargs).images[0]
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise torch.cuda.OutOfMemoryError(
            f"{exc}\n\n"
            "FLUX.2 ran out of GPU memory. On a 12 GB card the model does not fit "
            "fully resident, so use streaming CPU offload instead of full-GPU mode:\n"
            "  * GUI / CLI: add  --mode offload  (re-run install.bat and enter 8-11 "
            "for your VRAM to regenerate the launchers with offload), or\n"
            "  * shrink the generated region (fewer/larger tiles) so each run is "
            "smaller.\n"
            "Offload keeps only the active model component on the GPU and trades a "
            "little speed for fitting in 12 GB."
        ) from exc

    if mask is not None:
        if base_image is None:
            raise ValueError("A mask requires at least one reference image.")
        mask_img = mask if isinstance(mask, Image.Image) else Image.open(mask)
        mask_img = mask_img.convert("L").resize(base_image.size)
        edited = result.resize(base_image.size)
        result = Image.composite(edited, base_image, mask_img)

    return result


def main():
    parser = argparse.ArgumentParser(description="FLUX.2 local image editing / inpainting")
    parser.add_argument("--model-dir", required=True, help="Path to the downloaded model folder")
    parser.add_argument("--pipeline", choices=["klein", "dev"], default="klein")
    parser.add_argument("--mode", choices=["cuda", "offload", "quant"], default="cuda")
    parser.add_argument("--prompt", required=True, help="Text prompt / edit instruction")
    parser.add_argument(
        "--image",
        action="append",
        default=None,
        help="Reference image to edit. Repeat for multi-reference editing.",
    )
    parser.add_argument(
        "--mask",
        default=None,
        help="Optional grayscale mask (white = change). Requires --image.",
    )
    parser.add_argument("--output", default="output.png", help="Where to save the result")
    parser.add_argument("--steps", type=int, default=4, help="Number of denoising steps")
    parser.add_argument("--guidance", type=float, default=4.0, help="Guidance scale")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mask and not args.image:
        print("ERROR: --mask requires --image.")
        return 1

    print(f"Loading {args.pipeline} pipeline from {args.model_dir} (mode={args.mode})...")
    pipe = load_pipeline(args.model_dir, args.pipeline, args.mode)

    print("Running FLUX.2 inference...")
    result = generate_image(
        pipe,
        prompt=args.prompt,
        ref_images=args.image,
        mask=args.mask,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
    )

    result.save(args.output)
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
