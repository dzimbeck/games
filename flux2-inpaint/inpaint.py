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
import sys

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
            print(f"WARNING: 4-bit quantization unavailable ({exc}); using CPU offload.")
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

    return pipe


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(args.seed)

    ref_images = None
    base_image = None
    height, width = args.height, args.width
    if args.image:
        ref_images = [Image.open(p).convert("RGB") for p in args.image]
        base_image = ref_images[0]
        if height is None or width is None:
            width, height = round_to(base_image.width), round_to(base_image.height)

    call_kwargs = {
        "prompt": args.prompt,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance,
        "generator": generator,
    }
    if ref_images is not None:
        call_kwargs["image"] = ref_images
    if height is not None and width is not None:
        call_kwargs["height"] = height
        call_kwargs["width"] = width

    print("Running FLUX.2 inference...")
    result = pipe(**call_kwargs).images[0]

    # Masked inpainting: keep the original everywhere except the white mask area.
    if args.mask:
        mask = Image.open(args.mask).convert("L").resize(base_image.size)
        edited = result.resize(base_image.size)
        result = Image.composite(edited, base_image, mask)

    result.save(args.output)
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
