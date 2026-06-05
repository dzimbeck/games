"""Pure (no-torch) data model for the procedural map / tile grid.

The map is a ``cols`` x ``rows`` grid of tiles. Every tile is the same size in
pixels (``cell_w`` x ``cell_h``). A tile is either empty (treated as solid
black, i.e. "nothing has been painted here yet") or holds a ``PIL.Image``.

This module deliberately knows nothing about FLUX / torch so it can be reused
from the GUI, from automated scripts, or from a future HTTP API. The heavy
generation step lives in :mod:`inpaint`; this model just prepares the inputs it
needs and stitches the result back into the grid:

* :meth:`MapModel.region_bounds`  -> bounding box of a set of selected tiles.
* :meth:`MapModel.build_region_image` -> the current pixels of that box
  (filled tiles show through, empty tiles are black).
* :meth:`MapModel.build_region_mask`  -> a white-on-black mask marking the
  empty area that should be out/inpainted (white = generate here).
* :meth:`MapModel.apply_region_image`  -> slice a generated image back into the
  grid, one tile at a time.

Because out/inpainting is decided per pixel by the mask, a selection that is
fully empty becomes a pure outpaint (text-to-image with neighbour context),
while a selection that is partly filled becomes an inpaint that only changes the
remaining black area -- exactly the behaviour described in the design.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, List, Optional, Tuple

from PIL import Image

Cell = Tuple[int, int]
Bounds = Tuple[int, int, int, int]  # (c0, r0, c1, r1) inclusive


def round_to(value: float, multiple: int = 16) -> int:
    """Round ``value`` to the nearest positive ``multiple`` (min one multiple)."""
    return max(multiple, int(round(value / multiple) * multiple))


class MapModel:
    """A grid of equally sized image tiles."""

    def __init__(self, cols: int, rows: int, cell_w: int, cell_h: int):
        if cols < 1 or rows < 1:
            raise ValueError("Map must have at least one tile in each dimension.")
        # Cell size is always a multiple of 16 so generated tiles line up with
        # the model's latent grid and slice back cleanly.
        self.cols = int(cols)
        self.rows = int(rows)
        self.cell_w = round_to(cell_w)
        self.cell_h = round_to(cell_h)
        self._tiles: dict[Cell, Image.Image] = {}

    # ----- basic tile access -------------------------------------------------
    def in_bounds(self, c: int, r: int) -> bool:
        return 0 <= c < self.cols and 0 <= r < self.rows

    def is_empty(self, c: int, r: int) -> bool:
        """A tile is empty if nothing was painted or it is fully black."""
        img = self._tiles.get((c, r))
        if img is None:
            return True
        return not img.getbbox()  # getbbox() is None for an all-black image

    def get_tile(self, c: int, r: int) -> Image.Image:
        """Return the tile image, or a black tile for empty cells."""
        img = self._tiles.get((c, r))
        if img is None:
            return Image.new("RGB", (self.cell_w, self.cell_h), (0, 0, 0))
        return img

    def set_tile(self, c: int, r: int, img: Optional[Image.Image]) -> None:
        if not self.in_bounds(c, r):
            raise IndexError(f"Tile {(c, r)} is outside the {self.cols}x{self.rows} map.")
        if img is None:
            self._tiles.pop((c, r), None)
            return
        if img.mode != "RGB":
            img = img.convert("RGB")
        if img.size != (self.cell_w, self.cell_h):
            img = img.resize((self.cell_w, self.cell_h), Image.LANCZOS)
        self._tiles[(c, r)] = img

    def clear_tile(self, c: int, r: int) -> None:
        self._tiles.pop((c, r), None)

    # ----- region helpers ----------------------------------------------------
    @staticmethod
    def region_bounds(cells: Iterable[Cell]) -> Bounds:
        """Inclusive bounding box (c0, r0, c1, r1) covering ``cells``."""
        cells = list(cells)
        if not cells:
            raise ValueError("No tiles selected.")
        cs = [c for c, _ in cells]
        rs = [r for _, r in cells]
        return min(cs), min(rs), max(cs), max(rs)

    def region_pixel_size(self, bounds: Bounds) -> Tuple[int, int]:
        c0, r0, c1, r1 = bounds
        return (c1 - c0 + 1) * self.cell_w, (r1 - r0 + 1) * self.cell_h

    def build_region_image(self, bounds: Bounds) -> Image.Image:
        """Composite the current pixels of the bounding box into one image.

        Filled tiles show their content (which gives the generator surrounding
        context for continuity); empty tiles stay black.
        """
        c0, r0, c1, r1 = bounds
        width, height = self.region_pixel_size(bounds)
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if self.in_bounds(c, r) and not self.is_empty(c, r):
                    canvas.paste(self.get_tile(c, r),
                                 ((c - c0) * self.cell_w, (r - r0) * self.cell_h))
        return canvas

    def build_region_mask(self, bounds: Bounds, selected: Iterable[Cell]) -> Image.Image:
        """White (255) where we should generate, black (0) where we keep pixels.

        A pixel is generated when its tile is part of the selection *and* that
        tile is currently empty. Already-filled selected tiles and any
        unselected tiles inside the bounding box are preserved.
        """
        c0, r0, c1, r1 = bounds
        width, height = self.region_pixel_size(bounds)
        mask = Image.new("L", (width, height), 0)
        selected = set(selected)
        white = Image.new("L", (self.cell_w, self.cell_h), 255)
        for (c, r) in selected:
            if c0 <= c <= c1 and r0 <= r <= r1 and self.is_empty(c, r):
                mask.paste(white, ((c - c0) * self.cell_w, (r - r0) * self.cell_h))
        return mask

    def region_is_fully_empty(self, selected: Iterable[Cell]) -> bool:
        return all(self.is_empty(c, r) for c, r in selected)

    def apply_region_image(self, image: Image.Image, bounds: Bounds,
                           only: Optional[Iterable[Cell]] = None) -> List[Cell]:
        """Slice ``image`` (sized to ``bounds``) back into individual tiles.

        If ``only`` is given, just those tiles are written; otherwise every tile
        in the bounding box is updated. Returns the list of tiles that changed.
        """
        c0, r0, c1, r1 = bounds
        width, height = self.region_pixel_size(bounds)
        if image.size != (width, height):
            image = image.resize((width, height), Image.LANCZOS)
        if image.mode != "RGB":
            image = image.convert("RGB")
        only_set = set(only) if only is not None else None
        changed: List[Cell] = []
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if only_set is not None and (c, r) not in only_set:
                    continue
                if not self.in_bounds(c, r):
                    continue
                box = ((c - c0) * self.cell_w, (r - r0) * self.cell_h,
                       (c - c0 + 1) * self.cell_w, (r - r0 + 1) * self.cell_h)
                self.set_tile(c, r, image.crop(box))
                changed.append((c, r))
        return changed

    # ----- whole-map export / persistence -----------------------------------
    def render_full(self, scale: float = 1.0) -> Image.Image:
        """Render the entire map to a single image (optionally downscaled)."""
        full = Image.new("RGB", (self.cols * self.cell_w, self.rows * self.cell_h),
                         (0, 0, 0))
        for (c, r), img in self._tiles.items():
            full.paste(img, (c * self.cell_w, r * self.cell_h))
        if scale != 1.0:
            full = full.resize((max(1, int(full.width * scale)),
                                max(1, int(full.height * scale))), Image.LANCZOS)
        return full

    def save(self, path: str) -> None:
        """Save the map as a folder: ``meta.json`` plus one PNG per filled tile."""
        os.makedirs(path, exist_ok=True)
        meta = {
            "cols": self.cols,
            "rows": self.rows,
            "cell_w": self.cell_w,
            "cell_h": self.cell_h,
            "tiles": [],
        }
        for (c, r), img in self._tiles.items():
            name = f"tile_{c}_{r}.png"
            img.save(os.path.join(path, name))
            meta["tiles"].append({"c": c, "r": r, "file": name})
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "MapModel":
        with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        model = cls(meta["cols"], meta["rows"], meta["cell_w"], meta["cell_h"])
        for entry in meta.get("tiles", []):
            img = Image.open(os.path.join(path, entry["file"])).convert("RGB")
            model.set_tile(entry["c"], entry["r"], img)
        return model
