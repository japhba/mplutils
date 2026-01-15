import os
import uuid
from typing import Any, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import skunk

_ORIG_SAVEFIG = Figure.savefig


def _new_id(prefix: str = "svg") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _fig_store(fig: Figure) -> dict[str, str]:
    """Per-figure mapping from skunk-id -> svg (filepath or svg-string)."""
    store = getattr(fig, "_skunk_svg_replacements", None)
    if store is None:
        store = {}
        setattr(fig, "_skunk_svg_replacements", store)
    return store


def svgshow(
    self: Axes,
    svg: Union[str, os.PathLike],
    xy: Tuple[float, float],
    *,
    box_size: Tuple[float, float] = (50, 50),
    sk_id: Optional[str] = None,
    xycoords: str = "data",
    boxcoords: str = "offset points",
    xybox: Tuple[float, float] = (0, 0),
    frameon: bool = False,
    pad: float = 0.0,
    **annotationbbox_kwargs: Any,
):
    """
    Place an SVG as an "annotation box" anchored at `xy`.

    The SVG is preserved as vector *only* for SVG output (fig.savefig("*.svg")).
    For other formats, the placeholder box is what you'll see.
    """
    from matplotlib.offsetbox import AnnotationBbox

    if sk_id is None:
        sk_id = _new_id("box")

    box = skunk.Box(box_size[0], box_size[1], sk_id)

    ab = AnnotationBbox(
        box,
        xy,
        xybox=xybox,
        xycoords=xycoords,
        boxcoords=boxcoords,
        frameon=frameon,
        pad=pad,
        **annotationbbox_kwargs,
    )
    self.add_artist(ab)

    _fig_store(self.figure)[sk_id] = str(svg)
    return ab


def _patched_savefig(self: Figure, fname, *args, **kwargs):
    """
    If saving as SVG and this figure has svgshow replacements recorded,
    write post-processed SVG produced by skunk.insert(...).
    """
    fmt = kwargs.get("format", None)

    if fmt is None and isinstance(fname, (str, os.PathLike)):
        ext = os.path.splitext(str(fname))[1].lower()
        if ext == ".svg":
            fmt = "svg"

    store = getattr(self, "_skunk_svg_replacements", None)

    if fmt == "svg" and store:
        try:
            plt.figure(self.number)
        except Exception:
            pass

        out_svg = skunk.insert(dict(store))

        if hasattr(fname, "write"):
            try:
                fname.write(out_svg)
            except TypeError:
                fname.write(out_svg.encode("utf-8"))
            return

        with open(fname, "w", encoding="utf-8") as f:
            f.write(out_svg)
        return

    return _ORIG_SAVEFIG(self, fname, *args, **kwargs)


def patch_svgshow():
    """Install `ax.svgshow(...)` + SVG-save hook once."""
    if getattr(Axes, "_skunk_svgshow_patch_installed", False):
        return

    Axes.svgshow = svgshow
    Figure.savefig = _patched_savefig

    Axes._skunk_svgshow_patch_installed = True
