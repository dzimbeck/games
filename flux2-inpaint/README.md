# FLUX.2 Local Image Editing / Inpainting — one-click installer

A self-contained, "venv philosophy" installer for running [Black Forest Labs'
FLUX.2](https://github.com/black-forest-labs/flux2) image editing / inpainting
models locally on Windows. It mirrors the Qwen3-VL moderation installer: a single
`.bat` builds a **local** Python environment (pyenv-win + Python 3.11 + venv) inside
this folder, installs PyTorch + diffusers, and downloads the right model for your GPU.

## Quick start

1. Double-click **`install.bat`**.
2. When asked, enter your GPU **VRAM in GB** (just press Enter to accept the
   default of **12 GB**).
3. Wait for the environment to build and the model to download.
4. Run an edit:

   ```bat
   run_inpaint.bat --image input.png --prompt "make the sky a starry night" --output result.png
   ```

Everything is installed under `flux2-inpaint/ai-model/` (Python, the venv and the
model weights), so nothing touches your system Python.

### If the download is interrupted

The installer is **safe to re-run**. If your internet drops part-way through (a
common problem with multi-GB model downloads), just double-click `install.bat`
again:

- Steps that already finished are detected and skipped — pyenv, Python, the
  virtual environment and any Python packages that already import are left as-is.
- The model download is handled by **aria2** (driven through `aria2p`) for a
  consistent, always-moving progress line (percent, downloaded / total, speed,
  ETA). Each file is written straight into `ai-model/model/` with a small
  `<file>.aria2` control file beside it, so a dropped connection or a closed
  window **resumes from the exact byte it stopped at** — it never re-downloads
  data you already have (important if you pay per gigabyte) and never leaves
  giant hidden cache blobs. A `DOWNLOAD_COMPLETE` marker is written once the
  model is fully present so future runs skip the download entirely.

> **aria2 binary:** the installer uses a system `aria2c` if one is on your
> `PATH`; otherwise it downloads the correct prebuilt aria2 for your OS
> (Windows / Linux / macOS) from GitHub into `ai-model/.aria2` and reuses it.


## Grid map painter (GUI)

For bulk / procedural map building there is a simple grid "paint program",
`run_gui.bat` (created by the installer). It treats the world as a matrix of
equally sized tiles where **each tile is one FLUX.2 generation**:

1. On launch, choose the **matrix size** (number of tiles, e.g. `690 x 690`) and
   the **cell size** in pixels (e.g. `768 x 768`; rounded to a multiple of 16).
2. The view starts **zoomed out to 1/4** at the top-left tile so you can see the
   grid. Only the tiles inside the current viewport are rendered, so even very
   large maps stay responsive. Use `+` / `-` / mouse-wheel to zoom, the
   scrollbars or middle/right-drag to pan, and **Fit** to frame the whole map.
3. **Click** a tile to select it, or **drag** to select a block of adjacent
   tiles (click a selected tile again to deselect it).
4. Press **Generate** and type a description. Empty (black) tiles in the
   selection are **outpainted**; if part of the selection is already filled, the
   remaining black area is **inpainted** instead, using the filled neighbours for
   continuity. (This is decided per pixel by an automatic mask.)
5. **Upload reference** adds reference images, shown as thumbnails along the top;
   remove or swap them at any time. They are passed to the generator as extra
   guidance.
6. **Save map** writes the tiles to a folder; **Export PNG** flattens the whole
   map to a single image.

The first generation loads the model (it can take a while); afterwards the
pipeline stays in memory for fast subsequent tiles.

### Scripting / automation

The GUI is a thin layer over reusable, importable pieces so you can automate a
whole map or build an API later:

* `mapmodel.MapModel` — the pure (no-torch) tile grid: region compositing, mask
  building, slicing a result back into tiles, save/load.
* `inpaint.generate_image(...)` — one FLUX.2 generation (text-to-image, edit, or
  masked inpaint), shared by the CLI and the GUI.
* `mapgen.MapGenerator` — loads the pipeline once and fills a selection of tiles
  via `generate_region(model, cells, prompt, ref_images=...)`.

## Which model gets installed?

The installer asks for your VRAM and picks a FLUX.2 model + runtime mode. Model
choices follow the official [FLUX.2 "Which Model Should I
Use?"](https://github.com/black-forest-labs/flux2#which-model-should-i-use)
guidance:

| VRAM you enter | Model (Hugging Face repo) | Mode | License |
| --- | --- | --- | --- |
| 6–7 GB | `black-forest-labs/FLUX.2-klein-4B` | 4-bit quantized | Apache-2.0 |
| 8–11 GB | `black-forest-labs/FLUX.2-klein-4B` | CPU offload | Apache-2.0 |
| **12–15 GB (default)** | `black-forest-labs/FLUX.2-klein-4B` | full GPU | Apache-2.0 |
| 16–23 GB | `black-forest-labs/FLUX.2-klein-9B` | CPU offload | Non-commercial |
| 24–47 GB | `black-forest-labs/FLUX.2-klein-9B` | full GPU | Non-commercial |
| 48 GB+ | `black-forest-labs/FLUX.2-dev` | CPU offload | Non-commercial |

> **Licensing:** FLUX.2 [klein] **4B is Apache-2.0** and downloads without a login.
> FLUX.2 [klein] **9B** and FLUX.2 **[dev]** are gated, **non-commercial** models —
> accept the license on their Hugging Face page and run `huggingface-cli login`
> before installing.

## Usage

`run_inpaint.bat` (generated by the installer) already knows the model, mode and
default step count. Pass it any of the options below:

**Text-to-image**

```bat
run_inpaint.bat --prompt "a cat holding a sign that says hello world" --output cat.png
```

**Instruction editing** (FLUX.2 edits using your image as a reference)

```bat
run_inpaint.bat --image photo.png --prompt "turn the jacket bright red" --output edited.png
```

**Multi-reference editing**

```bat
run_inpaint.bat --image a.png --image b.png --prompt "combine these into one scene" --output combo.png
```

**Masked inpainting** (white in the mask = the area to change; the rest of the
original is preserved)

```bat
run_inpaint.bat --image room.png --mask mask.png --prompt "add a window on the wall" --output room_out.png
```

### All options

| Option | Description |
| --- | --- |
| `--prompt` | Text prompt / edit instruction (required) |
| `--image` | Reference image; repeat for multi-reference editing |
| `--mask` | Grayscale mask, white = change (requires `--image`) |
| `--output` | Output path (default `output.png`) |
| `--steps` | Denoising steps (distilled klein defaults to 4) |
| `--guidance` | Guidance scale (ignored by distilled models) |
| `--height`, `--width` | Output size (defaults to the input image size) |
| `--seed` | Random seed (default 42) |

## How it works / notes

- **About "inpainting":** FLUX.2 [klein] performs *instruction-based* editing using
  your input as a reference image rather than classic masked diffusion inpainting.
  When you pass `--mask`, this tool generates an edit and then composites it back
  onto the original so only the masked region changes.
- FLUX.2 pipelines (`Flux2KleinPipeline` / `Flux2Pipeline`) currently live on
  `diffusers` `main`, which is why the installer installs diffusers from Git.
- For a manual / non-Windows setup, see `requirements.txt` and install PyTorch with
  the CUDA build that matches your driver.

## Files

| File | Purpose |
| --- | --- |
| `install.bat` | One-click Windows installer (asks VRAM, builds venv, downloads model, writes `run_gui.bat` + `run_inpaint.bat`) |
| `download_model.py` | Downloads the selected FLUX.2 model snapshot locally with aria2 (resumable, visible progress) |
| `inpaint.py` | diffusers-based editing / inpainting runner (CLI + reusable `generate_image`) |
| `mapmodel.py` | Pure tile-grid model used by the GUI and for scripting/automation |
| `mapgen.py` | Bridges the grid model to FLUX.2 (`MapGenerator.generate_region`) |
| `map_gui.py` | Grid "paint program" GUI (Tkinter) for bulk out/inpainting |
| `requirements.txt` | Python dependencies for a manual setup |
