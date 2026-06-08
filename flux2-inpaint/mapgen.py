"""Bridge between the pure :class:`mapmodel.MapModel` grid and the FLUX.2
generator in :mod:`inpaint`.

This is the piece a script or future API would call to "fill in" part of the
map. It prepares the region image + mask from the model, runs one generation,
and writes the result back into the affected tiles.

Keeping this separate means :mod:`mapmodel` stays free of torch and the GUI can
import the model (and lay out the grid) before the heavy pipeline is loaded.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from PIL import Image

from inpaint import generate_image, load_pipeline
from mapmodel import Cell, MapModel


class MapGenerator:
    """Loads a FLUX.2 pipeline once and fills map regions on demand."""

    def __init__(self, model_dir: str, pipeline: str = "klein", mode: str = "cuda",
                 steps: int = 4, guidance: float = 4.0):
        self.model_dir = model_dir
        self.pipeline_kind = pipeline
        self.mode = mode
        self.steps = steps
        self.guidance = guidance
        self._pipe = None

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    def ensure_loaded(self) -> None:
        if self._pipe is None:
            self._pipe = load_pipeline(self.model_dir, self.pipeline_kind, self.mode)

    def generate_region(
        self,
        model: MapModel,
        selected: Sequence[Cell],
        prompt: str,
        ref_images: Optional[Iterable] = None,
        seed: int = 42,
    ) -> List[Cell]:
        """Out/inpaint the ``selected`` tiles of ``model`` from ``prompt``.

        * Fully-empty selection  -> outpaint (text-to-image with neighbour
          context from any filled tiles inside the bounding box).
        * Partly-filled selection -> inpaint: only the still-empty pixels of the
          selected tiles are changed; existing content is preserved.

        ``ref_images`` are optional extra reference images (paths or
        ``PIL.Image``) used for style/content guidance. Returns the changed
        tiles.
        """
        if not selected:
            raise ValueError("No tiles selected to generate.")
        self.ensure_loaded()

        bounds = model.region_bounds(selected)
        region = model.build_region_image(bounds)
        mask = model.build_region_mask(bounds, selected)
        width, height = model.region_pixel_size(bounds)

        # The region image (with its black holes) is always the first reference
        # so the generator sees the surrounding, already-painted context. Any
        # user-supplied references follow it.
        refs: List[Image.Image] = [region]
        if ref_images:
            for img in ref_images:
                refs.append(img if isinstance(img, Image.Image) else Image.open(img))

        result = generate_image(
            self._pipe,
            prompt=prompt,
            ref_images=refs,
            mask=mask,
            width=width,
            height=height,
            steps=self.steps,
            guidance=self.guidance,
            seed=seed,
        )

        # Only write the tiles the user selected; untouched neighbours that were
        # merely pulled in for context keep their original pixels.
        return model.apply_region_image(result, bounds, only=selected)
