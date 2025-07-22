import functools
import logging
import platform
import pprint
import subprocess
from contextlib import contextmanager
from pathlib import Path

from dataclasses import asdict

from matplotlib import cm
from matplotlib.ticker import Locator

import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.transforms import Affine2D, Bbox, TransformedBbox, blended_transform_factory
import numpy as np
import yaml

logger = logging.getLogger(__name__)

import os
import sys
import platform

def get_twin(ax, axis=['x', 'y']):
    _siblings = [getattr(ax, f"get_shared_{axis}_axes")().get_siblings(ax) for axis in axis]
    siblings = []
    for sib in _siblings:
        siblings.extend(sib)



    for sibling in siblings:
        if sibling.bbox.bounds == ax.bbox.bounds and sibling is not ax:
            return sibling 
    return None

def fractions(x,pos, step):
    if np.isclose((x/step)%(1./step),0.):
        # x is an integer, so just return that
        return '{:.0f}'.format(x)
    else:
        # this returns a latex formatted fraction
        return rf"${'+' if x > 0 else '-'}$"+'$\\frac{{{:2.0f}}}{{{:2.0f}}}$'.format(np.abs(x/step),1./step)
        # if you don't want to use latex, you could use this commented
        # line, which formats the fraction as "1/13"
        ### return '{:2.0f}/{:2.0f}'.format(x/step,1./step)



def get_inkscape_palettes_directory():
    if platform.system() == 'Windows':
        appdata = os.getenv('APPDATA')
        path = os.path.join(appdata, 'inkscape', 'palettes')
    elif platform.system() == 'Darwin':
        home = os.getenv('HOME')
        path = os.path.join(home, 'Library', 'Application Support', 'org.inkscape.Inkscape', 'config', 'inkscape', 'palettes')
    elif platform.system() == 'Linux':
        home = os.getenv('HOME')
        path = os.path.join(home, '.config', 'inkscape', 'palettes')
    else:
        raise Exception('Unsupported operating system.')
    
    return Path(path)



def plot_rasterplot(ax, Xt, norm=None, aspect="auto", extent=None,  axis_off=True, **largs):
    if extent is None:
        extent = [0, shp[1], 0, shp[0]]
    shp = Xt.shape
    cmap = largs.pop("cmap", "Greys")
    if norm is None:
        norm = plt.Normalize(vmin=-1, vmax=+1)
    mat = ax.matshow(Xt, cmap=cmap, norm=norm, aspect=aspect,
                     extent=extent, interpolation='none', rasterized=False)
    
    ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True, top=False, labeltop=False)
    ax.xaxis.set_ticks_position('bottom')

    if axis_off:
        # no ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # no ticklabels
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # spines to all sides
        ax.spines.top.set_visible(True)
        ax.spines.right.set_visible(True)
        ax.spines.bottom.set_visible(True)
        ax.spines.left.set_visible(True)



def fill_between(
        ax,
        x,
        y=None,
        y_mean=None,
        y_std=None,
        gauss_reduce=True,
        line=True,
        discrete=False,
        **line_args,
):
    if gauss_reduce:
        if y is not None:
            fac = y.shape[0] ** 0.5
        else:
            fac = gauss_reduce
    else:
        fac = 1

    fill_alpha = line_args.pop("fill_alpha", .3)
    line_alpha = line_args.pop("alpha", 1.)

    if (y is not None) and (y.shape[0] == 1):
        l, = ax.plot(x, y[0], alpha=line_alpha, **line_args)
        return l, None

    if y is not None:
        y = np.atleast_2d(y)
        if y.shape[0] == 1:
            # leave immediately
            l, = ax.plot(x, y[0], **line_args)
            return l, None
        mean = y.mean(axis=0)
        std = y.std(axis=0) / fac
        if (std < 1e-10).all(): logger.warning("Trivial std observed while attempting fill_between plot")
    else:
        mean = y_mean
        std = y_std / fac

    
    if not discrete:
        if line:
            line_args["markersize"] = 4
            (l,) = ax.plot(x, mean, alpha=line_alpha, **line_args)
            lc = l.get_color()
        else:
            l = None
            lc = None

        if (std != 0).any():
            c = line_args.get("color", lc)
            idx = np.argsort(x)
            x = np.array(x)
            fill = ax.fill_between(
                x[idx],
                (mean - std)[idx],
                (mean + std)[idx],
                alpha=fill_alpha,
                color=c,
                zorder=-10,
            )
        else:
            fill = None
    else:
        ls = line_args.pop("ls", "none")
        marker = line_args.pop("marker", "o")

        l = ax.errorbar(
            x, mean, yerr=std, ls=ls, fmt=marker, capsize=4, **line_args
        )
        fill = None

    return l, fill

def extra_scale_from_function(ax, func, label="", extra=False):
    ax_sigma = ax.twiny() if extra else ax
    ax_sigma.set_xlim(ax.get_xlim())
    locs = ax.get_xticks()
    vals = func(locs)
    ax_sigma.set_xticks(locs)
    ax_sigma.set_xticklabels([f"{val:.2f}" for val in vals])
    ax_sigma.set_xlabel(label)


def multiple_formatter(denominator=2, number=np.pi, latex=r"\pi"):
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den * x / number))
        com = gcd(num, den)
        (num, den) = (int(num / com), int(den / com))
        if den == 1:
            if num == 0:
                return r"$0$"
            if num == 1:
                return r"$%s$" % latex
            elif num == -1:
                return r"$-%s$" % latex
            else:
                return r"$%s%s$" % (num, latex)
        else:
            if num == 1:
                return r"$\frac{%s}{%s}$" % (latex, den)
            elif num == -1:
                return r"$\frac{-%s}{%s}$" % (latex, den)
            else:
                return r"$\frac{%s%s}{%s}$" % (num, latex, den)

    return _multiple_formatter

class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex=r"\pi"):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(
            multiple_formatter(self.denominator, self.number, self.latex)
        )


cm_ = 1 / 2.54


def init_mpl(tex=False):
    if tex:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Computer Modern"],
                "text.latex.preamble": r"\usepackage{amssymb}",
            }
        )
    plt.rcParams.update({"figure.dpi": 150})  # in point
    plt.rcParams.update({"font.size": 18})  # in point


def layout_ax(ax, tex=True):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


import matplotlib.pyplot as plt
from matplotlib.transforms import TransformedBbox, Bbox, Affine2D
import matplotlib.patches as patches

def tight_bbox(ax, debug=False):
    """
    Get the tight bounding box of an axes in figure coordinates.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to get the tight bounding box for.
    debug : bool
        If True, draw the tight bounding box on the figure.
        
    Returns:
    --------
    tight_bbox_fig : matplotlib.transforms.Bbox
        The tight bounding box in figure coordinates.
    """
    fig = ax.get_figure()
    renderer = fig.canvas.get_renderer()
    tight_bbox_raw = ax.get_tightbbox(renderer)
    tight_bbox_fig = TransformedBbox(tight_bbox_raw, fig.transFigure.inverted())
    
    if debug:
        # Draw the tight bounding box on the figure
        box_coords = tight_bbox_fig.get_points()
        x0, y0 = box_coords[0]
        width = box_coords[1, 0] - box_coords[0, 0]
        height = box_coords[1, 1] - box_coords[0, 1]
        
        # Create a rectangle patch for the tight bbox
        rect = patches.Rectangle(
            (x0, y0), width, height,
            linewidth=1, edgecolor='red', facecolor='none',
            linestyle='--', transform=fig.transFigure
        )
        
        # Get the current axes to properly overlay the rectangle
        fig.add_artist(rect)
        
        # Also draw the regular bbox for comparison
        regular_box = ax.get_position()
        regular_rect = patches.Rectangle(
            (regular_box.x0, regular_box.y0), 
            regular_box.width, regular_box.height,
            linewidth=1, edgecolor='blue', facecolor='none',
            linestyle=':', transform=fig.transFigure
        )
        fig.add_artist(regular_rect)
        
        # Add a legend to explain the colors
        fig.text(0.01, 0.01, 'Tight bbox (red), Regular bbox (blue)', 
                 color='black', fontsize=8, transform=fig.transFigure)
        
    return tight_bbox_fig

# Default label styling
LABEL_KWARGS = dict(size=12, weight='bold')

def make_ax_label(ax, label):
    """Add a label to an axes with default styling."""
    label_text = rf"$\mathbf{{{label.upper()}}}$"
    ax.text(x=-.2, y=1.2, s=label_text, transform=ax.transAxes, 
            fontsize=12, fontweight='bold', va='top', ha='right')

def add_panel_label(
    ax,
    letter,
    pad_x=5e-2,  # % fig x
    pad_y=1e-2,  # % fix x (sic!)
    use_tight_bbox=False,
    ha="right",
    va="bottom",
    transform=None,
    return_text=False,
    x=0.0,
    y=1.0,
    debug=False,
    **text_kwargs,
):
    """
    Add a panel label to an axes.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to add the label to.
    letter : str
        The label text.
    pad_x, pad_y : float
        Padding in the x and y directions.
    use_tight_bbox : bool
        Whether to use the tight bounding box for positioning.
    ha, va : str
        Horizontal and vertical alignment.
    transform : matplotlib.transforms.Transform
        Transform to use. If None, uses ax.transAxes.
    return_text : bool
        Whether to return the text object.
    x, y : float
        Position in axes coordinates.
    debug : bool
        If True, draw the tight bounding box on the figure.
    **text_kwargs : dict
        Additional text properties.
    """
    # Format the letter properly
    if "$" not in letter:
        letter = r"$\mathrm{\mathbf{" + letter + "}}$"
    
    # Use default kwargs if none provided
    if not text_kwargs:
        text_kwargs = LABEL_KWARGS.copy()
        
    assert pad_x >= 0 and pad_y >= 0, "Padding must be non-negative"
    assert pad_x < 1e-1 and pad_y < 1e-1, "Padding must be less than 0.1, as it's in fig_x coordinates!"
    
    # Use transAxes by default if no transform is provided
    transform = transform if transform is not None else ax.transAxes
    
    fig = ax.get_figure(root=True)
    if use_tight_bbox:
        # Get tight and regular bounding boxes
        tight_box = tight_bbox(ax, debug=debug)
        regular_box = ax.get_position()
        
        # Calculate offsets in axes coordinates for both x and y
        x_offset = (tight_box.x0 - regular_box.x0) / regular_box.width
        y_offset = (tight_box.y0 - regular_box.y0) / regular_box.height
        
        # Adjust the position based on the offsets and desired alignment
        if ha == "right" or ha == "center":
            adjusted_x = x - pad_x - x_offset
        else:  # "left"
            adjusted_x = x + pad_x - x_offset
            
        if va == "top" or va == "center":
            adjusted_y = y + pad_y - y_offset
        else:  # "bottom"
            adjusted_y = y - pad_y - y_offset
            
        # Debug: Mark the original and adjusted positions if debug is enabled
        if debug:
            # Original position
            ax.plot([x], [y], 'go', transform=transform, markersize=5)
            # Adjusted position
            ax.plot([adjusted_x], [adjusted_y], 'ro', transform=transform, markersize=5)
            # Add a legend
            ax.text(0.02, 0.02, 'Original (green), Adjusted (red)', 
                   color='black', fontsize=8, transform=ax.transAxes)
    else:
        # Use the provided coordinates directly
        trans_fig = blended_transform_factory(fig.transFigure, fig.transFigure)
        
        def trafo_tot(xy):
            xy_displ = trans_fig.transform(xy)
            xy_trafo = transform.inverted().transform(xy_displ)  # to trafo coordinates
            return xy_trafo
        
        pad_trafo = trafo_tot((pad_x, pad_y)) - trafo_tot((0, 0))
        adjusted_x = x - pad_trafo[0]
        adjusted_y = y + pad_trafo[1]

    
    # Create the text with the calculated position
    text = ax.text(
        adjusted_x,
        adjusted_y,
        letter,
        ha=ha,
        va=va,
        transform=transform,
        **text_kwargs
    )
    
    if return_text:
        return text
    
    
def format_angle(ax, n=2):
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / n))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter(n, np.pi)))



def xylabel_to_ticks(ax, which="both", pad=0.):
    fig = ax.get_figure()
    fig.canvas.draw()

    if which == "both":
        which = "all"

    if which == "x":
        which = "bottom"

    if which == "y":
        which = "left"

    if which == "all":
        for which_ in ["left", "bottom"]:
            xylabel_to_ticks(ax, which=which_, pad=pad)

    if which == "top" or which == "bottom":
        x_label = ax.xaxis.get_label()
        
        ax.xaxis.get_label().set_horizontalalignment("center")
        ax.xaxis.get_label().set_verticalalignment("bottom" if which == "top" else "top")
        ticklab = ax.xaxis.get_ticklabels()[0]
        trans = ticklab.get_transform()
        x_label_coords = trans.inverted().transform(ax.transAxes.transform(x_label.get_position()))

        ax.xaxis.set_label_coords(x_label_coords[0], (0 if which == "bottom" else 1) + pad, transform=trans)

    if which == "left" or which == "right":
        y_label = ax.yaxis.get_label()
        
        ax.yaxis.get_label().set_horizontalalignment("center")
        ax.yaxis.get_label().set_verticalalignment("bottom" if which == "left" else "top")
        ticklab = ax.yaxis.get_ticklabels()[0]
        trans = ticklab.get_transform()

        y_label_coords = trans.inverted().transform(ax.transAxes.transform(y_label.get_position()))
        ax.yaxis.set_label_coords((0 if which == "left" else 1) + pad, y_label_coords[1], transform=trans)


def frame_only(ax):
    no_spine(ax, which="top", spine=True)
    no_spine(ax, which="bottom", spine=True)
    no_spine(ax, which="left", spine=True)
    no_spine(ax, which="right", spine=True)

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch


class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    """Add an 3d arrow to an `Axes3D` instance."""

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, "arrow3D", _arrow3D)


def plot_arrow(ax):
    arrow_x = 0.  # x-coordinate of arrow tail
    arrow_y = 0.  # y-coordinate of arrow tail
    arrow_dx = 1.  # length of arrow along x-axis
    arrow_dy = 0  # length of arrow along y-axis

    ann1 = ax.annotate(
    "",  # empty label text
    xy=(arrow_x + arrow_dx, arrow_y + arrow_dy),  # endpoint of arrow
    xytext=(arrow_x, arrow_y),  # starting point of arrow
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),  # arrow properties
    )

    ann2 = ax.annotate(
        "$t$",  # label text
        xy=(arrow_x + arrow_dx, arrow_y + arrow_dy),  # endpoint of arrow
        xytext=(arrow_x + arrow_dx, arrow_y + arrow_dy),  # starting point of label
        ha="left",  # horizontal alignment of text
        va="center",  # vertical alignment of text
    )

    ax.axis("off")
    ax.set_xlim(0., 1.0)

def save_plot(
        path,
        configs={},
        fn_dict={},
        fig_or_ani=None,
        file_formats=None,
        use_hash=False,
        fn_prefix=None,
        include_filename=True,
        script_fn=None,
        **save_args
):
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    import yaml
    import hashlib
    import json

    # Merge configs
    config_ = {}
    for config in configs:
        try:
            config = asdict(config)
        except:
            pass
        config_ |= config

    # Default savefig args
    # save_args_dflt = dict(bbox_inches='tight')
    save_args_dflt = {}
    save_args = {**save_args_dflt, **save_args}

    # Ensure output directory exists
    Path(path).mkdir(parents=True, exist_ok=True)

    # Hash (using stdlib)
    if use_hash and config_:
        serialized = json.dumps(config_, sort_keys=True)
        config_hash = hashlib.sha1(serialized.encode('utf-8')).hexdigest()[:4]
    else:
        config_hash = ''

    # Script name
    if script_fn is None:
        script_fn = Path(__file__).stem

    # Build filename base
    name_parts = []
    if include_filename:
        name_parts.append(script_fn)
    if fn_dict:
        for k, v in fn_dict.items():
            name_parts.append(f"{str_to_fn(k)}_{str_to_fn(v)}")
    if config_hash:
        name_parts.append(config_hash)
    fn_base = "__".join(name_parts)[:100]
    if fn_prefix:
        fn_base = f"{fn_prefix}_{fn_base}"

    # Determine if fig_or_ani is a Figure or Animation
    is_animation = hasattr(fig_or_ani, 'save') and hasattr(fig_or_ani, 'frame_seq')

    # Determine file formats
    if not is_animation:
        if file_formats is None:
            file_formats = [plt.rcParams.get('savefig.format', 'png')]
    else:
        # animation: default to mp4
        file_formats = [fmt.lower() for fmt in (file_formats or ['mp4'])]

    # Save
    if not is_animation:
        fig = fig_or_ani or plt.gcf()
        for fmt in file_formats:
            if fmt == 'png':
                save_args.setdefault('dpi', 400)
            transparent = save_args.pop('transparent', True)
            fname = Path(path) / f"{fn_base}.{fmt}"
            fig.savefig(fname, transparent=transparent, **save_args)
    else:
        # set ffmpeg path if needed
        plt.rcParams['animation.ffmpeg_path'] = plt.rcParams.get('animation.ffmpeg_path', '')
        out_file = Path(path) / f"{fn_base}.{file_formats[0]}"
        fig_or_ani.save(str(out_file), **save_args)

    # Save config YAML
    def to_python_type(v):
        if np.isscalar(v):
            if np.issubdtype(type(v), np.integer): return int(v)
            if np.issubdtype(type(v), np.floating): return float(v)
        return v

    yml_path = Path(path) / f"{fn_base}.yml"
    with open(yml_path, 'w') as file:
        yaml.dump(
            {k: to_python_type(v) for k, v in config_.items()
             if not hasattr(v, '__len__') or isinstance(v, str)},
            file,
        )


def save_test_artifact(request, fig=None, title="", **kwargs):
    artifact_dir = TESTPATH / "artifacts" / request.node.path.stem
    artifact_dir.mkdir(exist_ok=True, parents=True)
    test_str = request.node.name.split('[')[0]
    if title != "": title = f"__{title}"
    config_str = request.node.callspec.id + title

    save_plot(
            artifact_dir,
            fig=fig,
            file_formats=["png"],
            fn_prefix=f"{test_str}__{config_str}",
            use_hash=True,
            include_filename=False,
            **kwargs
    )

def merge_lh(hl1, hl2):
    h1, l1 = hl1
    h2, l2 = hl2

    return (h1 + h2, l1 + l2)

def place_graphic(ax, inset_path, fit=None, mode="raster", inkscape_kwargs={}):
    import subprocess
    from pathlib import Path
    import platform
    fig = ax.get_figure()
    plt.rcParams['text.usetex'] = False
    ax.cla()
    ax.axis("off")
    # no_spine(ax, which="right", remove_all=True)

    # freeze fig to finish off layout, new in 3.6
    fig.canvas.draw()
    fig.set_layout_engine('none')

    ax_bbox = ax.get_position()
    fig_w, fig_h = fig.get_size_inches()

    plt.rcParams.update(
        {
            "pgf.texsystem": "lualatex",
            "pgf.preamble": r"\usepackage{graphicx}\usepackage[export]{adjustbox}\usepackage{amsmath}",
        }
    )

    from matplotlib.backends.backend_pgf import FigureCanvasPgf
    import matplotlib

    # TeX rendering does only work if saved as pdf
    matplotlib.backend_bases.register_backend("pdf", FigureCanvasPgf)

    bbox = {"width": ax_bbox.width * fig_w, "height": ax_bbox.height * fig_h}

    import tempfile, shutil, os

    def create_temporary_copy(path):

        temp_dir = Path(
            "/tmp" if platform.system() == "Darwin" else tempfile.gettempdir()
        )
        rand_seq = np.random.choice(["a", "b", "c", "d", "e"], size=10)
        temp_path = os.path.join(temp_dir, f'{"".join(rand_seq)}{path.suffix}')
        shutil.copy2(path, temp_path)
        return temp_path, temp_dir

    path_str, temp_dir = create_temporary_copy(inset_path)
    if inset_path.suffix == ".svg":
        # we first need to convert to a format that allows us to embed
        inkscape_kwargs_dflt = {}
        if mode == "raster":
            inkscape_kwargs_dflt = inkscape_kwargs_dflt | {'export-dpi': 300}
            path_str_rendered = str(temp_dir / (inset_path.stem + ".png"))
            command = [
                "inkscape",
                path_str,
                f"--export-filename={path_str_rendered}",
            ]
        else:
            path_str_rendered = str(temp_dir / (inset_path.stem + ".pdf"))
            command = [
                "inkscape",
                path_str,
                f"--export-filename={path_str_rendered}"
            ]
        command.extend([f"--{key}" if value is None else f"--{key}={value}" for key, value in inkscape_kwargs.items()])

        p = subprocess.run(command, capture_output=True, text=True)
        path_str = path_str_rendered

        # print inkscape --help if failed
        if p.returncode != 0:
            print(p.stderr)
            help_out = subprocess.run(["inkscape", "--help"], capture_output=True, text=True)
            print(help_out.stdout)
            
    if fit is None:
        w, h = get_w_h(inset_path)
        if w / h > bbox["width"] / bbox["height"]:
            fit = "width"
        else:
            fit = "height"
    else:
        assert "width" in fit or "height" in fit
        
    if path_str.endswith(".pdf"):
        # embed via LaTeX – quite buggy!
        tex_cmd = ""
        tex_cmd += r"\centering"
        tex_cmd += rf"\includegraphics[{fit}={{{bbox[fit]:.5f}in}}]{{{path_str}}}"
        print(bbox[fit])
        ax.text(0.0, 0.0, tex_cmd)
    else:
        # embed via imshow
        ax.imshow(plt.imread(path_str))

def plot_traj(ax, x, y, alpha_min=.3, alpha_max=1., mark='none', gain=None, **largs):
    if gain is None:
       gain = lambda t: t

    c = largs.pop('c', largs.pop('color', 'k'))
    alpha = largs.pop('alpha', 1)
    # convert to RGBA
    
    c = mcolors.to_rgb(c) if type(c) == str else c
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    gains = gain(np.linspace(0, 1, len(x)))
    alphas = alpha_min + (alpha_max - alpha_min) * gains

    ls = largs.pop('ls', '-')
    cmap = ListedColormap([(c, alpha) for alpha in alphas])
    if ls != 'none':
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1))
        color_val = x / x.max()
        # Set the values used for colormapping
        lc.set_array(color_val)
        lc.set_linewidth(1)
        line = ax.add_collection(lc)
    else:
        line = None

    # make markers at start and end
    m_args = dict(marker='o', zorder=10, ls='none')
    if 'start' in mark.lower():
        ax.plot(x[0], y[0], c=c, alpha=alpha_min, **(m_args | largs))
    if 'end' in mark.lower():
        ax.plot(x[-1], y[-1], c=c, alpha=alpha_max,  **(m_args | largs))

    if 'all' in mark.lower():
        for i in range(len(x)):
            ax.plot(x[i], y[i], c=c, alpha=alphas[i], **(m_args | largs))

    return line,

def color_ax(ax, color):
    ax.yaxis.label.set_color(color)
    ax.spines["right"].set_edgecolor(color)
    ax.spines["left"].set_edgecolor(color)
    ax.tick_params(axis="y", colors=color)


def N_ticks(ax, N=2, which="x", axis_end=False):
    getattr(ax, f"{which}axis").set_major_locator(plt.MaxNLocator(N - 1))
    if axis_end:
        if N > 2: raise NotImplementedError
        lims = getattr(ax, f"get_{which}lim")()
        dlim = np.ptp(lims)
        margins = ax.margins()
        getattr(ax, f"set_{which}ticks")([lims[0] + dlim * margins[0], lims[1] - dlim * margins[1]])


@contextmanager
def no_autoscale(ax=None, axis="both"):
    ax = ax or plt.gca()
    ax.figure.canvas.draw()
    lims = [ax.get_xlim(), ax.get_ylim()]
    yield
    if axis == "both" or axis == "x":
        ax.set_xlim(*lims[0])
    if axis == "both" or axis == "y":
        ax.set_ylim(*lims[1])

def match_scale(ax1, ax2, which):
    min1, max1 = getattr(ax1, f"get_{which}lim")()
    min2, max2 = getattr(ax2, f"get_{which}lim")()

    min12 = min(min1, min2)
    max12 = max(max1, max2)

    getattr(ax1, f"set_{which}lim")(min12, max12)
    getattr(ax2, f"set_{which}lim")(min12, max12)



# https://matplotlib.org/stable/gallery/subplots_axes_and_figures/axes_zoom_effect.html
def zoom_effect(axparent, axchild, xmin, xmax, pad_top=None, pad_bottom=None, **kwargs):
    """
    Connect *ax1* and *ax2*. The *xmin*-to-*xmax* range in both axes will
    be marked.

    Parameters
    ----------
    axparent
        The main axes.
    axchild
        The zoomed axes.
    xmin, xmax
        The limits of the colored area in both plot axes.
    **kwargs
        Arguments passed to the patch constructor.
    """

    from matplotlib.transforms import Bbox, TransformedBbox
    from mpl_toolkits.axes_grid1.inset_locator import (
        BboxPatch,
        BboxConnector,
        BboxConnectorPatch,
    )

    def connect_bbox(
            bbox1, bbox2, loc1a, loc2a, loc1b, loc2b, prop_lines, prop_patches=None
    ):
        if prop_patches is None:
            prop_patches = {
                **prop_lines,
                "alpha": prop_lines.get("alpha", 1) * 0.2,
                "clip_on": False,
            }

        c1 = BboxConnector(
            bbox1, bbox2, loc1=loc1a, loc2=loc2a, clip_on=False, **prop_lines
        )
        c2 = BboxConnector(
            bbox1, bbox2, loc1=loc1b, loc2=loc2b, clip_on=False, **prop_lines
        )

        bbox_patch1 = BboxPatch(bbox1, **prop_patches)
        bbox_patch2 = BboxPatch(bbox2, **prop_patches)

        p = BboxConnectorPatch(
            bbox1,
            bbox2,
            # loc1a=3, loc2a=2, loc1b=4, loc2b=1,
            loc1a=loc1a,
            loc2a=loc2a,
            loc1b=loc1b,
            loc2b=loc2b,
            clip_on=False,
            **{k: v for k, v in prop_patches.items() if k != "color"},
        )

        return c1, c2, bbox_patch1, bbox_patch2, p

    bbox = Bbox.from_extents(xmin, -pad_bottom, xmax, 1 + pad_top)

    bbox_prnt = TransformedBbox(bbox, axparent.get_xaxis_transform())
    bbox_chld = TransformedBbox(bbox, axchild.get_xaxis_transform())

    prop_patches = {**kwargs, "ec": "none", "alpha": 0.2}

    c1, c2, bbox_patch1, bbox_patch2, p = connect_bbox(
        bbox_prnt,
        bbox_chld,
        loc1a=3,
        loc1b=4,
        loc2a=2,
        loc2b=1,
        prop_lines=kwargs,
        prop_patches=prop_patches,
    )

    axparent.add_patch(bbox_patch1)
    # axchild.add_patch(bbox_patch2)
    axchild.add_patch(c1)
    axchild.add_patch(c2)
    axchild.add_patch(p)

    return c1, c2, bbox_patch1, bbox_patch2, p


import matplotlib.colors as mcolors


class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max([np.abs(self.vmin), np.abs(self.vmax)])
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def plot_grids(axd, grids, Q0, T, independent_noise=None, downsample_factor=1, grid_thy=None):
    fig = list(axd.values())[0].figure

    grid_h, grid_x, grid_q, grid_err = grids
    grids_h, grids_q, grids_err = [grid_h] * int(T - 2), [grid_q] * int(T - 2), [grid_err] * int(T - 2)

    Tc =  grid_h.shape[0] - T
    xx_ini = grid_h[Tc, Tc]

    # slc = slice(int(Tc - 100), None, None)
    slc = slice(None, None, None)

    import matplotlib
    norm_err = mcolors.SymLogNorm(vmin=1e-15, vmax=1, linthresh=1e-5)
    cmap_err = matplotlib.cm.Reds.copy()
    cmap_err.set_bad("red", 1.0)

    extent = (-0.5, grid_h.shape[0] - 0.5, -0.5, grid_h.shape[0] - 0.5)
    def update(ti, axd, grids):
        for k in axd:
            if k != "discr" and k in ["h_corr", "q_corr"]:
                axd[k].cla()
                try:
                    cbar1.remove()
                    cbar2.remove()
                except:
                    pass
                dat_mat = grids[k][ti][slc, slc]

                cmap_corr = matplotlib.cm.Greys.copy()
                norm_corr = mcolors.SymLogNorm(vmin=np.nanmin(dat_mat), vmax=1.5, linthresh=1e-15)
                cmap_corr.set_bad("red", 1.0)
                
                mat1 = axd[k].imshow(dat_mat, cmap=cmap_corr, norm=norm_corr, interpolation="none")
                mat2 = axd[k + "_s"].imshow(
                    (dat_mat - (dat_mat + dat_mat.T) / 2),
                    cmap=cmap_corr,
                    norm=norm_corr,
                    interpolation="none")

                cbar1 = fig.colorbar(mat1, ax=axd[k], shrink=1.)
                cbar2 = fig.colorbar(mat2, ax=axd[k + "_s"], shrink=1.)

    cmap_corr = matplotlib.cm.Greys.copy()
    norm_corr = mcolors.SymLogNorm(vmin=np.nanmin(grid_thy[slc, slc]), vmax=1.5, linthresh=1e-15)
    cmap_corr.set_bad("red", 1.0)
    if grid_thy is not None:
        axd["h_thy"].imshow(grid_thy[slc, slc], cmap=cmap_corr, interpolation="none", norm=norm_corr,)
        axd["discr_thy_rel"].imshow(np.abs(grid_thy[slc, slc] - grid_h[slc, slc])/np.abs(grid_thy[slc, slc]), cmap=cmap_err, interpolation="none", norm=norm_err,)

    # discrepancy_PDE = verify_PDE(
    #     Q=grids_h[-1], Q0=net.Q0, Tc=Tc, T=T, act_func_slope=net.act_func_slope, dt=dt, g=g, D=D,
    #     independent_noise=independent_noise, downsample_factor=downsample_factor
    # )
    # where_high = np.where((discrepancy_PDE > 1e-2) & (~np.isnan(discrepancy_PDE)))


    mat = axd["discr_rel"].imshow(grid_err[slc, slc],
                              cmap=cmap_err, norm=norm_err, interpolation="none"
                              )
    fig.colorbar(mat, ax=axd["discr_rel"], shrink=1.)


    frames = np.arange(len(grids_h))
    # anim = animation.FuncAnimation(
    #     fig,
    #     update,
    #     fargs=(axd, dict(h_corr=grids_h, x_corr=grids_x)),
    #     frames=frames,
    # )
    update(frames[-1], axd, dict(h_corr=grids_h, q_corr=grids_q))

    for k, ax in axd.items():
        # ax.axvline(Tc, ymin=0., ymax=1., lw=0.1, ls="dashed", alpha=0.3, zorder=+50)
        # ax.axhline(Tc, xmin=0., xmax=1., lw=0.1, ls="dashed", alpha=0.3, zorder=+50)
        # ax.axvline(0, ymin=0., ymax=1., lw=0.1, ls="dashed", alpha=0.3, zorder=+50)
        # ax.axhline(0, xmin=0., xmax=1., lw=0.1, ls="dashed", alpha=0.3, zorder=+50)

        ax.set_title(ax._label)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.spines.bottom.set_visible(False)
        ax.spines.left.set_visible(False)

def no_spine(
        ax, which, label=False, spine=False, ticklabel=False, ticks=False, remove_all=False
):
    if which == "bottom":
        whichxy = "x"
    elif which == "left":
        whichxy = "y"
    else:
        whichxy = "xy"

    if remove_all:
        label = False
        ticklabel = False
        ticks = False
        spine = False

    kwargs = {"which": "both", which: ticks, f"label{which}": ticklabel}  # major/minor
    getattr(ax.spines, which).set_visible(spine)

    for wxy in whichxy:
        getattr(ax, f"set_{wxy}label", "" if not label else False)
        ax.tick_params(axis=wxy, **kwargs)


from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation


class Annotation3D(Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self, s, xy=(0, 0), *args, **kwargs)
        self._verts3d = xyz

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy = (xs, ys)
        Annotation.draw(self, renderer)


def get_w_h(path):
    if path.suffix == ".svg":
        import xml.etree.ElementTree as ET

        svg = ET.parse(path).getroot().attrib
        import re

        w = svg["width"]
        h = svg["height"]
        w = float(re.sub("[^0-9]", "", w))
        h = float(re.sub("[^0-9]", "", h))
    elif path.suffix == ".pdf":
        from PyPDF2 import PdfFileReader

        input1 = PdfFileReader(open(path, "rb"))
        mediaBox = input1.getPage(0).mediaBox
        w, h = mediaBox.getWidth(), mediaBox.getHeight()
    else:
        raise NotImplementedError
    return w, h

def joint_title(axes, subfig, title, **text_kwargs):
    # get joint bounding box
    xmin, xmax = +np.inf, -np.inf
    ymin, ymax = +np.inf, -np.inf

    for ax in np.reshape(axes, (-1,)):
        bbox = ax.get_position()
        xmin = np.min([bbox.xmin, xmin])
        xmax = np.max([bbox.xmax, xmax])

        ymin = np.min([bbox.ymin, ymin])
        ymax = np.max([bbox.ymax, ymax])

    x_fig = (xmax+xmin)/2
    y_fig = ymax

    assert 0<x_fig <1
    assert 0<y_fig <1
    text = subfig.text(
            s=title,
            ha="center",
            va="bottom",
            x=x_fig,
            y=y_fig,
            zorder=100,
            # transform=blended_transform_factory(fig.transFigure, fig.transFigure),
            **text_kwargs,
        )

def square_widths(w, h, width_ratios, height_ratios=None, square_idxs=(0,0), leave="height"):
    """
    Takes in width and height of a figure and the width ratios, and returns ratios such that the first panel [0,0] will have equal aspect.
    """
    # prepend list axis
    square_idxs = np.atleast_2d(square_idxs)
    i,j = square_idxs[0]
    width_ratios = np.array(width_ratios)/np.sum(width_ratios)
    width_ratios_out = width_ratios.copy()
    height_ratios = np.array(height_ratios)/np.sum(height_ratios)
    height_ratios_out = height_ratios.copy()
    assert np.allclose(np.sum(height_ratios),1)
    assert np.allclose(np.sum(width_ratios),1)

    if (len(height_ratios) == 1 and not leave== "width") or (len(width_ratios) == 1 and not leave== "height"):
        raise ValueError

    if leave == "width":
        h_square_in = height_ratios[i] * h
        w_square_in = h_square_in

        w_square_rat = w_square_in / w
        w_sum_omit = np.sum(width_ratios_out[np.arange(len(width_ratios)) != j])
        width_ratios_out = width_ratios_out / w_sum_omit
        width_ratios_out *= (1-w_square_rat)
        assert np.allclose(np.sum(width_ratios_out[np.arange(len(width_ratios)) != j]),1-w_square_rat)
        width_ratios_out[j] = w_square_rat
        assert np.allclose(np.sum(width_ratios_out),1)
    elif leave == "height":
        w_square_in = width_ratios[i] * w
        h_square_in = w_square_in

        h_square_rat = h_square_in / h
        h_sum_omit = np.sum(height_ratios_out[np.arange(len(height_ratios)) != i])
        height_ratios_out = height_ratios_out / h_sum_omit
        height_ratios_out *= (1-h_square_rat)
        assert np.allclose(np.sum(height_ratios_out[np.arange(len(height_ratios)) != i]),1-h_square_rat)
        height_ratios_out[i] = h_square_rat
        assert np.allclose(np.sum(height_ratios_out),1)
    else:
        raise ValueError

    assert np.allclose(height_ratios_out[i]*h, width_ratios_out[j]*w)

    if len(square_idxs) > 1:
        if leave == "height":
            w_out = w_square_in * len(width_ratios)
            h_out = h
        elif leave == "width":
            w_out = w
            h_out = h_square_in * len(height_ratios)
        else:
            raise ValueError
    else:
        w_out, h_out = w, h
    return width_ratios_out, height_ratios_out, w_out, h_out

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import colorsys

def lighten_colormap(cmap, rate=0.5):
    cmap = plt.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    
    # Convert RGBA colors to HLS and increase lightness
    colors_hls = np.array([colorsys.rgb_to_hls(*color[:3]) for color in colors])
    colors_hls[:,1] = colors_hls[:,1] * (1 + rate)
    colors_hls[:,1][colors_hls[:,1] > 1] = 1 # limit lightness to 1

    colors[:,:3] = np.array([colorsys.hls_to_rgb(*color_hls) for color_hls in colors_hls])
    
    return mcolors.LinearSegmentedColormap.from_list(cmap.name + "_light", colors, cmap.N)

def alpha_colormap(cmap, alpha=0.5):
    # Add alpha channel to the colormap
    rgba_colors = np.zeros((256, 4))
    rgba_colors[:, :3] = cmap(np.arange(256)/256)[:,:3]
    rgba_colors[:, 3] = alpha

    # Create a new colormap from the resulting RGBA array
    cmap = mcolors.LinearSegmentedColormap.from_list(cmap.name + "_light", rgba_colors)

    return cmap
        

def fill_between(
        ax,
        x,
        y=None,
        y_mean=None,
        y_std=None,
        gauss_reduce=False,
        line=True,
        discrete=False,
        **line_args,
):
    if gauss_reduce:
        if y is not None:
            fac = y.shape[0] ** 0.5
        else:
            fac = gauss_reduce
    else:
        fac = 1
        
    alpha = line_args.pop("alpha", .3)
    fill_alpha = line_args.pop("fill_alpha", .3) if 'fill_alpha' in line_args else .3
    line_alpha = line_args.pop("line_alpha", 1.) if 'line_alpha' in line_args else alpha

    if (y is not None) and (y.shape[0] == 1):
        l, = ax.plot(x, y[0], **line_args)
        return l, None

    if y is not None:
        y = np.atleast_2d(y)
        if y.shape[0] == 1:
            # leave immediately
            l, = ax.plot(x, y[0], **line_args)
            return l, None
        mean = y.mean(axis=0)
        std = y.std(axis=0) / fac
        if (std < 1e-10).all(): logger.warning("Trivial std observed while attempting fill_between plot")
    else:
        mean = y_mean
        std = y_std / fac

    
    if not discrete:
        if line:
            (l,) = ax.plot(x, mean, alpha=line_alpha, **line_args)
        else:
            l = None

        if (std != 0).any():
            c = line_args.get("color", l.get_color())
            idx = np.argsort(x)
            x = np.array(x)
            fill = ax.fill_between(
                x[idx],
                (mean - std)[idx],
                (mean + std)[idx],
                alpha=fill_alpha,
                color=c,
                zorder=-10,
            )
        else:
            fill = None
    else:
        ls = line_args.pop("ls", "none")
        marker = line_args.pop("marker", "o")

        l = ax.errorbar(
            x, mean, yerr=std, ls=ls, fmt=marker, capsize=4, **line_args
        )
        fill = None

    return l, fill


def sym_lims(ax, which="y"):
    # get y-axis limits of the plot
    low, high = getattr(ax, f"get_{which}lim")()
    # find the new limits
    bound = max(abs(low), abs(high))
    # set new limits
    getattr(ax, f"set_{which}lim")(-bound, bound)


def multiple_formatter(denominator=2, number=np.pi, latex=r"\pi"):
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den * x / number))
        com = gcd(num, den)
        (num, den) = (int(num / com), int(den / com))
        if den == 1:
            if num == 0:
                return r"$0$"
            if num == 1:
                return r"$%s$" % latex
            elif num == -1:
                return r"$-%s$" % latex
            else:
                return r"$%s%s$" % (num, latex)
        else:
            if num == 1:
                return r"$\frac{%s}{%s}$" % (latex, den)
            elif num == -1:
                return r"$\frac{-%s}{%s}$" % (latex, den)
            else:
                return r"$\frac{%s%s}{%s}$" % (num, latex, den)

    return _multiple_formatter


class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex=r"\pi"):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(
            multiple_formatter(self.denominator, self.number, self.latex)
        )


def add_tick(ax, loc, label, which="x", keeplims=True, minor=True):
    # raise NotImplementedError("Does strange things to axes scalings...")
    lim = getattr(ax, f"get_{which}lim")()

    axis = getattr(ax, f"get_{which}axis")()

    xt = list(getattr(ax, f"get_{which}ticks")(minor=minor))

    from matplotlib.ticker import FixedFormatter
    from matplotlib.ticker import FixedLocator

    majorminor = 'major' if not minor else 'minor'
    # make fixed if not
    if not isinstance(getattr(axis, f"get_{majorminor}_formatter"), FixedFormatter):
        ax.figure.canvas.draw()
        getattr(axis, f"set_{majorminor}_locator")(FixedLocator(xt))
        formatter = getattr(axis, f"get_{majorminor}_formatter")()

        xtl = [formatter(xt_) for xt_ in xt]
        getattr(axis, f"set_{majorminor}_formatter")(FixedFormatter(xtl))
    else:
        xtl = list(getattr(ax, f"get_{which}ticklabels")(minor=minor))

    # xtl = [mpltxt.get_text() for mpltxt in xtl]

    axis.remove_overlapping_locs = False
    locs = np.atleast_1d(loc)
    labels = np.atleast_1d(label)
    locs = list(locs)
    labels = list(labels)

    getattr(ax, f"set_{which}ticks")(xt + locs, xtl + labels, minor=minor)
    if keeplims:
        getattr(ax, f"set_{which}lim")(lim)


def plot_norm(ax, net, sim_opts, Xt=None, norms=None, warmup=False, N_SIGMA=None):
    if Xt is not None and norms is None:
        assert Xt.ndim == 3
        norms = np.linalg.norm(Xt, axis=-1) ** 2 / net.N
    elif Xt is None and norms is not None:
        pass
    else:
        raise ValueError
    norms_mean = norms.mean(axis=0)
    norms_std = norms.std(axis=0) * N_SIGMA
    ts = sim_opts.ts if not warmup else sim_opts.ts_c
    l, fill = fill_between(ax, x=ts, y_mean=norms_mean, y_std=norms_std, line=True,
                           color="C0" if sim_opts.field == "x" else "C1")
    ax.set_ylabel(r"$|X|^2/N$")
    ax.axhline(
        net.Q0 / net.g2,
        label=r"$x\sim Q_0/g^2$",
        color="C0",
        ls="dashed",
        lw=3,
    )
    ax.axhline(
        net.Q0,
        label=r"$h\sim Q_0$",
        color="C1",
        ls="dashed",
        lw=3,
    )

    ax.set_ylim(bottom=0)


class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """
    def __init__(self, linthresh):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically
        """
        self.linthresh = linthresh

    def __call__(self):
        'Return the locations of the ticks'
        majorlocs = self.axis.get_majorticklocs()

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i-1]
            if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                ndivs = 10
            else:
                ndivs = 9
            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))


"""
From https://matplotlib.org/stable/users/explain/customizing.html
https://seaborn.pydata.org/generated/seaborn.plotting_context.html#seaborn.plotting_context
https://github.com/mwaskom/seaborn/blob/f0b48e891a1bb573b7a46cfc9936dcd35d7d4f24/seaborn/rcmod.py#L335
"""
_style_keys = [

    "axes.facecolor",
    "axes.edgecolor",
    "axes.grid",
    "axes.axisbelow",
    "axes.labelcolor",

    "figure.facecolor",

    "grid.color",
    "grid.linestyle",

    "text.color",

    "xtick.color",
    "ytick.color",
    "xtick.direction",
    "ytick.direction",
    "lines.solid_capstyle",

    "patch.edgecolor",
    "patch.force_edgecolor",

    "image.cmap",
    "font.family",
    "font.sans-serif",

    "xtick.bottom",
    "xtick.top",
    "ytick.left",
    "ytick.right",

    "axes.spines.left",
    "axes.spines.bottom",
    "axes.spines.right",
    "axes.spines.top",

]

_context_keys = [

    "font.size",
    "axes.labelsize",
    "axes.titlesize",
    "xtick.labelsize",
    "ytick.labelsize",
    "legend.fontsize",
    "legend.title_fontsize",

    "axes.linewidth",
    "grid.linewidth",
    "lines.linewidth",
    "lines.markersize",
    "patch.linewidth",

    "xtick.major.width",
    "ytick.major.width",
    "xtick.minor.width",
    "ytick.minor.width",

    "xtick.major.size",
    "ytick.major.size",
    "xtick.minor.size",
    "ytick.minor.size",

]


def set_context(context=None, font_scale=1, rc=None):
    """
    Set the parameters that control the scaling of plot elements.

    These parameters correspond to label size, line thickness, etc.
    Calling this function modifies the global matplotlib `rcParams`. For more
    information, see the :doc:`aesthetics tutorial <../tutorial/aesthetics>`.

    The base context is "notebook", and the other contexts are "paper", "talk",
    and "poster", which are version of the notebook parameters scaled by different
    values. Font elements can also be scaled independently of (but relative to)
    the other values.

    See :func:`plotting_context` to get the parameter values.

    Parameters
    ----------
    context : dict, or one of {paper, notebook, talk, poster}
        A dictionary of parameters or the name of a preconfigured set.
    font_scale : float, optional
        Separate scaling factor to independently scale the size of the
        font elements.
    rc : dict, optional
        Parameter mappings to override the values in the preset seaborn
        context dictionaries. This only updates parameters that are
        considered part of the context definition.

    Examples
    --------

    .. include:: ../docstrings/set_context.rst

    """
    context_object = plotting_context(context, font_scale, rc)
    mpl.rcParams.update(context_object)

class _RCAesthetics(dict):
    def __enter__(self):
        rc = mpl.rcParams
        self._orig = {k: rc[k] for k in self._keys}
        self._set(self)

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._set(self._orig)

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper

class _PlottingContext(_RCAesthetics):
    """Light wrapper on a dict to set context temporarily."""
    _keys = _context_keys
    _set = staticmethod(set_context)



def plotting_context(context=None, font_scale=1, rc=None, return_dict=True):
    """
    Get the parameters that control the scaling of plot elements.

    These parameters correspond to label size, line thickness, etc. For more
    information, see the :doc:`aesthetics tutorial <../tutorial/aesthetics>`.

    The base context is "notebook", and the other contexts are "paper", "talk",
    and "poster", which are version of the notebook parameters scaled by different
    values. Font elements can also be scaled independently of (but relative to)
    the other values.

    This function can also be used as a context manager to temporarily
    alter the global defaults. See :func:`set_theme` or :func:`set_context`
    to modify the global defaults for all plots.

    Parameters
    ----------
    context : None, dict, or one of {paper, notebook, talk, poster}
        A dictionary of parameters or the name of a preconfigured set.
    font_scale : float, optional
        Separate scaling factor to independently scale the size of the
        font elements.
    rc : dict, optional
        Parameter mappings to override the values in the preset seaborn
        context dictionaries. This only updates parameters that are
        considered part of the context definition.

    Examples
    --------

    .. include:: ../docstrings/plotting_context.rst

    """
    if context is None:
        context_dict = {k: mpl.rcParams[k] for k in _context_keys}

    elif isinstance(context, dict):
        context_dict = context

    else:

        contexts = ["paper", "notebook", "talk", "poster"]
        if context not in contexts and type(context) not in [int, float]:
            raise ValueError(f"context must be in {', '.join(contexts)}")

        # Set up dictionary of default parameters
        texts_base_context = {

            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "legend.title_fontsize": 12,

        }

        base_context = {

            "axes.linewidth": 1.25,
            "grid.linewidth": 1,
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            "patch.linewidth": 1,

            "xtick.major.width": 1.25,
            "ytick.major.width": 1.25,
            "xtick.minor.width": 1,
            "ytick.minor.width": 1,

            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.minor.size": 4,
            "ytick.minor.size": 4,

        }
        base_context.update(texts_base_context)

        # Scale all the parameters by the same factor depending on the context
        scale = dict(paper=.8, notebook=1, talk=1.5, poster=2)[context] if type(context) == str else context
        context_dict = {k: v * scale for k, v in base_context.items()}

        # Now independently scale the fonts
        font_keys = texts_base_context.keys()
        font_dict = {k: context_dict[k] * font_scale for k in font_keys}
        context_dict.update(font_dict)



    # Wrap in a _PlottingContext object so this can be used in a with statement
    context_object = _PlottingContext(context_dict)

    return context_dict if return_dict else context_object





def c_line(ax, x, y, c, cmap, **largs):
    import matplotlib.collections as mcoll

    # Create a set of line segments
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(c.min(), c.max())
    lc = mcoll.LineCollection(segments, cmap=cmap, norm=norm, **largs)

    # Set the values used for colormapping
    lc.set_array(c)
    line = ax.add_collection(lc)

    ax.dataLim.y0 = y.min()
    ax.dataLim.y1 = y.max()
    ax.autoscale_view()


# Define mosaic layout using subfigures
def create_subfig_mosaic(parent_fig, mosaic):
    """Create a dictionary of subfigures based on mosaic layout"""
    rows = len(mosaic)
    cols = max(len(row) for row in mosaic)
    
    # Create grid of subfigures
    subfigs_grid = parent_fig.subfigures(rows, cols)
    if rows == 1:
        subfigs_grid = [subfigs_grid]
    if cols == 1:
        subfigs_grid = [[sf] for sf in subfigs_grid]
    elif rows == 1:
        subfigs_grid = [subfigs_grid]
    
    # Map labels to subfigures
    subfig_dict = {}
    for i, row in enumerate(mosaic):
        for j, label in enumerate(row):
            if label != '.':  # Skip empty cells
                subfig_dict[label] = subfigs_grid[i][j]
    
    return subfig_dict