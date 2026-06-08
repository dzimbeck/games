"""Procedural map painter -- a simple grid GUI for bulk out/inpainting.

Think "tile paint program": the map is a grid of equally sized cells. Each cell
is one FLUX.2 generation. You click (or drag) to select one or more adjacent
tiles, press **Generate**, type a description, and the empty (black) parts of the
selection are out/inpainted with continuity from neighbouring tiles. Reference
images can be uploaded and are shown as thumbnails along the top.

Design notes
------------
* On launch you choose the matrix size (tiles) and the cell size (pixels). The
  cell size is rounded to a multiple of 16.
* The view starts zoomed out (1/4) at the top-left tile (1, 1) so the whole map
  is visible; only the tiles inside the current viewport are rendered.
* The heavy work is split out: :mod:`mapmodel` is the pure grid, :mod:`mapgen`
  drives FLUX.2. The model loads lazily on the first generation so the window
  opens instantly.

Run it via ``run_gui.bat`` (written by ``install.bat``) or directly::

    python map_gui.py --model-dir ai-model/model --pipeline klein --mode cuda
"""

from __future__ import annotations

import argparse
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image, ImageTk

from mapmodel import Cell, MapModel, round_to

START_ZOOM = 0.25  # zoomed out 1/4 so the whole map is visible to begin with
MIN_ZOOM = 0.02
MAX_ZOOM = 4.0
GRID_COLOR = "#333333"
SELECT_OUTLINE = "#39c0ff"
SELECT_FILL = "#39c0ff"
THUMB_SIZE = 96


class StartupDialog(simpledialog.Dialog):
    """Asks for the matrix (tile) size and the per-cell pixel size."""

    def body(self, master):
        self.title("New map")
        ttk.Label(master, text="Matrix size (number of tiles):").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 2))
        ttk.Label(master, text="Columns").grid(row=1, column=0, sticky="e")
        ttk.Label(master, text="Rows").grid(row=2, column=0, sticky="e")
        self.cols = ttk.Entry(master)
        self.rows = ttk.Entry(master)
        self.cols.insert(0, "690")
        self.rows.insert(0, "690")
        self.cols.grid(row=1, column=1, sticky="w")
        self.rows.grid(row=2, column=1, sticky="w")

        ttk.Label(master, text="Cell size (pixels, rounded to x16):").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(8, 2))
        ttk.Label(master, text="Width").grid(row=4, column=0, sticky="e")
        ttk.Label(master, text="Height").grid(row=5, column=0, sticky="e")
        self.cw = ttk.Entry(master)
        self.ch = ttk.Entry(master)
        self.cw.insert(0, "768")
        self.ch.insert(0, "768")
        self.cw.grid(row=4, column=1, sticky="w")
        self.ch.grid(row=5, column=1, sticky="w")
        return self.cols

    def validate(self) -> bool:
        try:
            cols = int(self.cols.get())
            rows = int(self.rows.get())
            cw = round_to(float(self.cw.get()))
            ch = round_to(float(self.ch.get()))
        except (ValueError, TypeError):
            messagebox.showerror("Invalid input", "Please enter whole numbers.")
            return False
        if cols < 1 or rows < 1:
            messagebox.showerror("Invalid input", "Matrix size must be at least 1x1.")
            return False
        self.result = (cols, rows, cw, ch)
        return True


class MapApp:
    def __init__(self, root: tk.Tk, model: MapModel, generator_factory):
        self.root = root
        self.model = model
        self._generator_factory = generator_factory
        self._generator = None

        self.zoom = START_ZOOM
        self.selection: Set[Cell] = set()
        self.ref_paths: List[str] = []
        self._ref_thumbs: List[ImageTk.PhotoImage] = []
        self._photo_cache: Dict[Cell, Tuple[int, int, int, ImageTk.PhotoImage]] = {}
        self._tile_version: Dict[Cell, int] = {}
        self._busy = False
        self._drag_start: Optional[Cell] = None
        self._dragging = False
        self._preview_rect: Optional[Tuple[Cell, Cell]] = None

        root.title("Procedural Map Painter")
        root.geometry("1100x800")
        self._build_ui()
        self.root.after(50, self.redraw)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        toolbar = ttk.Frame(self.root, padding=4)
        toolbar.pack(side="top", fill="x")

        ttk.Button(toolbar, text="Generate", command=self.on_generate).pack(side="left")
        ttk.Button(toolbar, text="Clear selection",
                   command=self.clear_selection).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Erase tiles",
                   command=self.erase_selection).pack(side="left", padx=2)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(toolbar, text="Upload reference",
                   command=self.upload_reference).pack(side="left")
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(toolbar, text="-", width=3,
                   command=lambda: self.set_zoom(self.zoom / 1.25)).pack(side="left")
        ttk.Button(toolbar, text="+", width=3,
                   command=lambda: self.set_zoom(self.zoom * 1.25)).pack(side="left")
        ttk.Button(toolbar, text="Fit", command=self.fit_zoom).pack(side="left", padx=2)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(toolbar, text="Save map", command=self.save_map).pack(side="left")
        ttk.Button(toolbar, text="Export PNG",
                   command=self.export_png).pack(side="left", padx=2)

        # Reference thumbnail strip.
        self.ref_strip = ttk.Frame(self.root, padding=(4, 2))
        self.ref_strip.pack(side="top", fill="x")
        self.ref_placeholder = ttk.Label(
            self.ref_strip, text="No reference images. Use 'Upload reference'.",
            foreground="#888888")
        self.ref_placeholder.pack(side="left")

        # Scrollable canvas grid.
        body = ttk.Frame(self.root)
        body.pack(side="top", fill="both", expand=True)
        self.canvas = tk.Canvas(body, bg="#101010", highlightthickness=0)
        hbar = ttk.Scrollbar(body, orient="horizontal", command=self.canvas.xview)
        vbar = ttk.Scrollbar(body, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self._on_scroll_x,
                              yscrollcommand=self._on_scroll_y)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        self._hbar, self._vbar = hbar, vbar

        # Status bar.
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(self.root, textvariable=self.status, relief="sunken",
                  anchor="w", padding=2).pack(side="bottom", fill="x")

        self._update_scrollregion()

        # Events.
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        # Pan with the middle/right button.
        for btn in ("2", "3"):
            self.canvas.bind(f"<ButtonPress-{btn}>",
                             lambda e: self.canvas.scan_mark(e.x, e.y))
            self.canvas.bind(f"<B{btn}-Motion>",
                             lambda e: (self.canvas.scan_dragto(e.x, e.y, gain=1),
                                        self.redraw()))
        # Mouse-wheel zoom (Windows/macOS and Linux).
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self.set_zoom(self.zoom * 1.1))
        self.canvas.bind("<Button-5>", lambda e: self.set_zoom(self.zoom / 1.1))

    # ----------------------------------------------------------- geometry
    @property
    def disp_w(self) -> int:
        return max(1, int(round(self.model.cell_w * self.zoom)))

    @property
    def disp_h(self) -> int:
        return max(1, int(round(self.model.cell_h * self.zoom)))

    def _update_scrollregion(self):
        self.canvas.configure(scrollregion=(
            0, 0, self.model.cols * self.disp_w, self.model.rows * self.disp_h))

    def _on_scroll_x(self, *args):
        self._hbar.set(*args)
        self.redraw()

    def _on_scroll_y(self, *args):
        self._vbar.set(*args)
        self.redraw()

    def _on_wheel(self, event):
        self.set_zoom(self.zoom * (1.1 if event.delta > 0 else 1 / 1.1))

    def event_to_cell(self, event) -> Optional[Cell]:
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        c = int(x // self.disp_w)
        r = int(y // self.disp_h)
        if self.model.in_bounds(c, r):
            return (c, r)
        return None

    # --------------------------------------------------------------- zoom
    def set_zoom(self, zoom: float):
        zoom = max(MIN_ZOOM, min(MAX_ZOOM, zoom))
        if abs(zoom - self.zoom) < 1e-6:
            return
        self.zoom = zoom
        self._photo_cache.clear()  # display size changed; rebuild thumbnails
        self._update_scrollregion()
        self.redraw()

    def fit_zoom(self):
        cw = self.canvas.winfo_width() or 1
        ch = self.canvas.winfo_height() or 1
        zx = cw / (self.model.cols * self.model.cell_w)
        zy = ch / (self.model.rows * self.model.cell_h)
        self.set_zoom(min(zx, zy))
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    # ------------------------------------------------------------ drawing
    def _bump_tile(self, cell: Cell):
        self._tile_version[cell] = self._tile_version.get(cell, 0) + 1
        self._photo_cache.pop(cell, None)

    def _tile_photo(self, c: int, r: int) -> Optional[ImageTk.PhotoImage]:
        """Return a cached display-sized PhotoImage for a filled tile."""
        if self.model.is_empty(c, r):
            return None
        version = self._tile_version.get((c, r), 0)
        cached = self._photo_cache.get((c, r))
        if cached and cached[0] == self.disp_w and cached[1] == self.disp_h \
                and cached[2] == version:
            return cached[3]
        img = self.model.get_tile(c, r).resize((self.disp_w, self.disp_h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self._photo_cache[(c, r)] = (self.disp_w, self.disp_h, version, photo)
        return photo

    def _visible_range(self) -> Tuple[int, int, int, int]:
        x0 = self.canvas.canvasx(0)
        y0 = self.canvas.canvasy(0)
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        c0 = max(0, int(x0 // self.disp_w))
        r0 = max(0, int(y0 // self.disp_h))
        c1 = min(self.model.cols - 1, int((x0 + w) // self.disp_w))
        r1 = min(self.model.rows - 1, int((y0 + h) // self.disp_h))
        return c0, r0, c1, r1

    def redraw(self):
        self.canvas.delete("all")
        dw, dh = self.disp_w, self.disp_h
        c0, r0, c1, r1 = self._visible_range()
        # Render only the tiles inside the viewport (plus the edge tiles).
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                x, y = c * dw, r * dh
                photo = self._tile_photo(c, r)
                if photo is not None:
                    self.canvas.create_image(x, y, image=photo, anchor="nw")
                else:
                    self.canvas.create_rectangle(x, y, x + dw, y + dh,
                                                 fill="#000000", outline=GRID_COLOR)
                if (c, r) in self.selection:
                    self.canvas.create_rectangle(
                        x + 1, y + 1, x + dw - 1, y + dh - 1,
                        outline=SELECT_OUTLINE, width=2,
                        fill=SELECT_FILL, stipple="gray25")
        # Grid lines only when tiles are large enough to be useful.
        if dw >= 6 and dh >= 6:
            for c in range(c0, c1 + 2):
                self.canvas.create_line(c * dw, r0 * dh, c * dw, (r1 + 1) * dh,
                                        fill=GRID_COLOR)
            for r in range(r0, r1 + 2):
                self.canvas.create_line(c0 * dw, r * dh, (c1 + 1) * dw, r * dh,
                                        fill=GRID_COLOR)
        # Drag-selection preview.
        if self._preview_rect:
            (pc0, pr0), (pc1, pr1) = self._preview_rect
            self.canvas.create_rectangle(
                min(pc0, pc1) * dw, min(pr0, pr1) * dh,
                (max(pc0, pc1) + 1) * dw, (max(pr0, pr1) + 1) * dh,
                outline=SELECT_OUTLINE, width=2, dash=(4, 2))
        self._update_status()

    def _update_status(self):
        if self._busy:
            return
        sel = len(self.selection)
        self.status.set(
            f"Map {self.model.cols}x{self.model.rows} tiles | "
            f"cell {self.model.cell_w}x{self.model.cell_h}px | "
            f"zoom {self.zoom*100:.0f}% | selected: {sel}")

    # --------------------------------------------------------- selection
    def on_press(self, event):
        cell = self.event_to_cell(event)
        self._drag_start = cell
        self._dragging = False

    def on_drag(self, event):
        if self._drag_start is None:
            return
        cell = self.event_to_cell(event)
        if cell is None:
            return
        self._dragging = True
        self._preview_rect = (self._drag_start, cell)
        self.redraw()

    def on_release(self, event):
        if self._drag_start is None:
            return
        if self._dragging and self._preview_rect:
            (c0, r0), (c1, r1) = self._preview_rect
            for r in range(min(r0, r1), max(r0, r1) + 1):
                for c in range(min(c0, c1), max(c0, c1) + 1):
                    self.selection.add((c, r))
        else:
            cell = self.event_to_cell(event)
            if cell is not None:
                if cell in self.selection:
                    self.selection.discard(cell)
                else:
                    self.selection.add(cell)
        self._drag_start = None
        self._dragging = False
        self._preview_rect = None
        self.redraw()

    def clear_selection(self):
        self.selection.clear()
        self.redraw()

    def erase_selection(self):
        if not self.selection:
            return
        for cell in self.selection:
            self.model.clear_tile(*cell)
            self._bump_tile(cell)
        self.redraw()

    # -------------------------------------------------------- references
    def upload_reference(self):
        paths = filedialog.askopenfilenames(
            title="Choose reference image(s)",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All", "*.*")])
        for p in paths:
            if p not in self.ref_paths:
                self.ref_paths.append(p)
        self._rebuild_ref_strip()

    def _rebuild_ref_strip(self):
        for child in self.ref_strip.winfo_children():
            child.destroy()
        self._ref_thumbs.clear()
        if not self.ref_paths:
            ttk.Label(self.ref_strip,
                      text="No reference images. Use 'Upload reference'.",
                      foreground="#888888").pack(side="left")
            return
        for path in list(self.ref_paths):
            frame = ttk.Frame(self.ref_strip, padding=2)
            frame.pack(side="left", padx=2)
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
                thumb = ImageTk.PhotoImage(img)
                self._ref_thumbs.append(thumb)
                ttk.Label(frame, image=thumb).pack()
            except Exception:  # noqa: BLE001 - show name even if preview fails
                ttk.Label(frame, text="(unreadable)").pack()
            name = os.path.basename(path)
            ttk.Label(frame, text=(name[:14] + "...") if len(name) > 16 else name,
                      font=("TkDefaultFont", 7)).pack()
            ttk.Button(frame, text="remove", width=7,
                       command=lambda p=path: self._remove_ref(p)).pack()

    def _remove_ref(self, path: str):
        if path in self.ref_paths:
            self.ref_paths.remove(path)
        self._rebuild_ref_strip()

    # -------------------------------------------------------- generation
    def on_generate(self):
        if self._busy:
            return
        if not self.selection:
            messagebox.showinfo("Generate", "Select one or more tiles first.")
            return
        partial = not self.model.region_is_fully_empty(self.selection)
        verb = "Inpaint" if partial else "Outpaint"
        prompt = simpledialog.askstring(
            f"{verb} {len(self.selection)} tile(s)",
            "Describe what should appear in the selected area:")
        if not prompt:
            return
        self._busy = True
        self.status.set(f"{verb}ing {len(self.selection)} tile(s)... "
                        "(first run loads the model, please wait)")
        selected = list(self.selection)
        refs = list(self.ref_paths)
        threading.Thread(target=self._run_generation,
                         args=(selected, prompt, refs), daemon=True).start()

    def _run_generation(self, selected, prompt, refs):
        try:
            if self._generator is None:
                self._generator = self._generator_factory()
            changed = self._generator.generate_region(
                self.model, selected, prompt, ref_images=refs or None)
            self.root.after(0, self._on_generation_done, changed, None)
        except Exception as exc:  # noqa: BLE001 - report to the user
            self.root.after(0, self._on_generation_done, None, exc)

    def _on_generation_done(self, changed, error):
        self._busy = False
        if error is not None:
            messagebox.showerror("Generation failed", str(error))
            self.status.set("Generation failed.")
            return
        for cell in changed or []:
            self._bump_tile(cell)
        self.redraw()
        self.status.set(f"Done. Updated {len(changed or [])} tile(s).")

    # -------------------------------------------------------------- files
    def save_map(self):
        path = filedialog.askdirectory(title="Save map to an (empty) folder")
        if not path:
            return
        self.model.save(path)
        self.status.set(f"Saved map to {path}")

    def export_png(self):
        path = filedialog.asksaveasfilename(
            title="Export whole map as PNG", defaultextension=".png",
            filetypes=[("PNG", "*.png")])
        if not path:
            return
        # Large maps are huge at full resolution; cap the long edge.
        full_long = max(self.model.cols * self.model.cell_w,
                        self.model.rows * self.model.cell_h)
        scale = min(1.0, 8192 / full_long)
        self.model.render_full(scale=scale).save(path)
        self.status.set(f"Exported map to {path}")


def make_generator_factory(args):
    """Defer importing torch/mapgen until the first generation is requested."""
    def factory():
        from mapgen import MapGenerator
        return MapGenerator(args.model_dir, pipeline=args.pipeline,
                            mode=args.mode, steps=args.steps, guidance=args.guidance)
    return factory


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Procedural map painter (GUI)")
    parser.add_argument("--model-dir",
                        default=os.path.join(here, "ai-model", "model"),
                        help="Path to the downloaded FLUX.2 model folder")
    parser.add_argument("--pipeline", choices=["klein", "dev"], default="klein")
    parser.add_argument("--mode", choices=["cuda", "offload", "quant"], default="cuda")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--cell-w", type=int, default=None)
    parser.add_argument("--cell-h", type=int, default=None)
    args = parser.parse_args()

    root = tk.Tk()

    if None in (args.cols, args.rows, args.cell_w, args.cell_h):
        dialog = StartupDialog(root)
        if not getattr(dialog, "result", None):
            return 0
        cols, rows, cw, ch = dialog.result
    else:
        cols, rows, cw, ch = args.cols, args.rows, args.cell_w, args.cell_h

    model = MapModel(cols, rows, cw, ch)
    MapApp(root, model, make_generator_factory(args))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
