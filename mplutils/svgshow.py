import os
import uuid
import logging
from typing import Any, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import skunk

logger = logging.getLogger("mplutils.svgshow")

_ORIG_SAVEFIG = Figure.savefig
_IN_SKUNK_SAVE = False


def _root_figure(fig: Figure) -> Figure:
    root = fig
    while True:
        parent = getattr(root, "figure", None)
        if parent is None or parent is root:
            break
        root = parent
    return root


def _axes_box_size_points(ax: Axes) -> Tuple[float, float]:
    root_fig = _root_figure(ax.figure)
    canvas = root_fig.canvas
    if canvas is None:
        root_fig.canvas.draw()
        canvas = root_fig.canvas
    try:
        renderer = canvas.get_renderer()
    except Exception:
        root_fig.canvas.draw()
        renderer = canvas.get_renderer()
    bbox = ax.get_window_extent(renderer=renderer)
    dpi = root_fig.get_dpi()
    return (bbox.width * 72.0 / dpi, bbox.height * 72.0 / dpi)


def _new_id(prefix: str = "svg") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _fig_store(fig: Figure) -> dict[str, str]:
    """Per-figure mapping from skunk-id -> svg (filepath or svg-string)."""
    fig = _root_figure(fig)
    store = getattr(fig, "_skunk_svg_replacements", None)
    if store is None:
        store = {}
        setattr(fig, "_skunk_svg_replacements", store)
    return store


def svgshow(
    self: Axes,
    svg: Union[str, os.PathLike],
    xy: Tuple[float, float] = (0.5, 0.5),
    *,
    box_size: Optional[Tuple[float, float]] = (50, 50),
    sk_id: Optional[str] = None,
    xycoords: str = "axes fraction",
    boxcoords: str = "offset points",
    xybox: Tuple[float, float] = (0, 0),
    frameon: bool = False,
    pad: float = 0.0,
    off: bool = True,
    fill: bool = True,
    scale: float = 1.0,
    **annotationbbox_kwargs: Any,
):
    """
    Place an SVG as an "annotation box" anchored at `xy`.

    If `fill` is True, the SVG scales to the full axes extent.
    Use `scale` to expand or shrink the box (e.g., 1.1).

    The SVG is preserved as vector *only* for SVG output (fig.savefig("*.svg")).
    For other formats, the placeholder box is what you'll see.
    """
    from matplotlib.offsetbox import AnnotationBbox

    if isinstance(svg, os.PathLike):
        svg_path = os.fspath(svg)
        if not os.path.exists(svg_path):
            raise FileNotFoundError(f"SVG file not found: {svg_path}")
    elif isinstance(svg, str):
        stripped = svg.lstrip()
        if "<svg" not in svg and not stripped.startswith("<?xml"):
            if not os.path.exists(svg):
                raise FileNotFoundError(f"SVG file not found: {svg}")

    if fill:
        box_size = _axes_box_size_points(self)
        box_size = (box_size[0] * scale, box_size[1] * scale)

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

    if off:
        self.axis("off")

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

    global _IN_SKUNK_SAVE
    if _IN_SKUNK_SAVE:
        return _ORIG_SAVEFIG(self, fname, *args, **kwargs)

    if fmt == "svg" and store:
        try:
            plt.figure(self.number)
        except Exception:
            pass

        _IN_SKUNK_SAVE = True
        try:
            out_svg = skunk.insert(dict(store))
        finally:
            _IN_SKUNK_SAVE = False

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
        return False

    Axes.svgshow = svgshow
    Figure.savefig = _patched_savefig

    Axes._skunk_svgshow_patch_installed = True
    logger.info("mplutils.svgshow patch installed")
    return True


if patch_svgshow():
    print("mplutils.svgshow patch installed")


if __name__ == "__main__":
    from pathlib import Path

    patch_svgshow()

    test_svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
        '<circle cx="5" cy="5" r="4" fill="red"/></svg>'
    )

    fig, axes = plt.subplots(2, 3, figsize=(6, 4))
    for ax in axes.ravel():
        ax.plot([0, 1], [0, 1])

    axes[0, 0].svgshow(test_svg, (0.5, 0.5))
    axes[0, 1].svgshow(test_svg, (0.5, 0.5))
    axes[0, 2].svgshow(test_svg, (0.5, 0.5), off=False)

    downloads_dir = Path.home() / "Downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    out_path = downloads_dir / "svgshow_test.svg"
    fig.savefig(out_path)

    output = out_path.read_text(encoding="utf-8")

    assert "<svg" in output and "circle" in output, "svgshow insert failed"
    print(f"svgshow __main__ test passed (saved to {out_path})")
