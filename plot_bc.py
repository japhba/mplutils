import logging
import platform
import pprint
import subprocess
from contextlib import contextmanager
from pathlib import Path

from dataclasses import asdict

from matplotlib import cm
from matplotlib.ticker import Locator
from bayesianchaos.CONST import TESTPATH


import joblib
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D, Bbox, TransformedBbox, blended_transform_factory
from bayesianchaos import MON_SIZE, show_plot
import numpy as np
import yaml

if "SHOW" not in globals(): from bayesianchaos.CONST import SHOW
logger = logging.getLogger(__name__)

import os
import sys
import platform

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

def pt_to_data_coords(ax, pt):
    dpi = ax.figure.dpi  # Get the figure's DPI
    display_spacing = pt * dpi / 72  # Convert pt to display spacing
    trans = ax.transData.inverted()  # Get the inverted data transformation object

    # Calculate the corresponding data spacing
    data_spacing = abs(trans.transform((display_spacing, 0))[0] - trans.transform((0, 0))[0])
    return data_spacing


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





def plot_kernel(ax, k, projection="xx", align_labels=True, bar=True, xx=None, thetas=None, order="1-xx", grid=False,p = None, **largs):

    if (xx is None) and (thetas is None):
        xx = 1-np.geomspace(1e-5,1,100)
        xx = np.concatenate([[1], xx])
    elif (xx is None) and (thetas is not None):
        xx = np.cos(thetas)
    elif (xx is not None) and (thetas is None):
        pass
    else:
        raise ValueError


    if projection == "xx":
        x_resc = 1-xx if order == "1-xx" else xx
        ax.set_xlim(-.1, 1.1)
        ax.set_xticks([0,1])

        if bar is False:
            ax.set_xlabel(rf"${'1-' if order == '1-xx' else ''}\boldsymbol{{x}} \cdot \boldsymbol{{x}}'$",  math_fontfamily='cm')
        else:
            ax.set_xlabel(rf"${'1-' if order == '1-xx' else ''}\bar{{\boldsymbol{{x}}}} \cdot \bar{{\boldsymbol{{x}}}}'$",  math_fontfamily='cm')
        ax.set_aspect(1)

    elif projection == "theta":
        x_resc = np.arccos(xx) if order == "1-xx" else np.pi/2 - np.arccos(xx)
        ax.set_xlim(0, np.pi/2*1.1)
        ax.set_xticks([-.1*np.pi/2, np.pi/2])
        ax.set_xticklabels([0,r"$\frac{\pi}{2}$"],  math_fontfamily='cm')
        ax.set_xlabel(r"$\Delta\theta$",  math_fontfamily='cm')
        dxlim = np.diff(ax.get_xlim())
        dylim = np.diff(ax.get_ylim())
        ax.set_aspect(dxlim/dylim)
    else:
        raise ValueError

    if p is not None:
        from bayesianchaos.scripts.colors import GRAYS
        # rescale colormap
        from matplotlib.colors import ListedColormap
        cmap = GRAYS
        cs = cmap(np.linspace(.5,1,500))
        cmap = ListedColormap(cs)
        c_line(ax, x_resc, k(xx), c=p(xx), cmap=cmap, **largs)
        l = None
    else:
        l, = ax.plot(x_resc, k(xx), **largs)

    ax.set_ylim(-.1, 1.1)
    # ax.set_title(f"g={g:.2f}")

    print(plt.rcParams['text.usetex'])
    print(plt.rcParams['mathtext.fontset'])
    
    ax.set_yticks([0,1])
    if bar is False:
        ax.set_ylabel(r"$\boldsymbol{\phi}_{\boldsymbol{x}}\cdot\boldsymbol{\phi}_{\boldsymbol{x}'}$", math_fontfamily='cm')
    else:
        ax.set_ylabel(r"$ \bar{\boldsymbol{\phi}}_{\boldsymbol{x}} \cdot \bar{\boldsymbol{\phi}}_{\boldsymbol{x}'}$", math_fontfamily='cm')

    if align_labels:
        xylabel_to_ticks(ax, which="both")

    if grid:
        ax.spines.right.set_visible(True)
        ax.spines.top.set_visible(True)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.set_xticks([.25, .5, .75], minor=True)
        ax.set_yticks([.25, .5, .75], minor=True)
        ax.grid(which='major')
        ax.grid(which='minor', alpha=0.2, linewidth=0.5)

    return l,


def extra_scale_from_function(ax, func, label="", extra=False):
    ax_sigma = ax.twiny() if extra else ax
    ax_sigma.set_xlim(ax.get_xlim())
    locs = ax.get_xticks()
    vals = func(locs)
    ax_sigma.set_xticks(locs)
    ax_sigma.set_xticklabels([f"{val:.2f}" for val in vals])
    ax_sigma.set_xlabel(label)


def multiple_formatter(denominator=2, number=np.pi, latex="\pi"):
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


def plot_autocorr(x, ts, net=None):
    skip = 10

    autocorrs = (
        np.atleast_2d(np.einsum("...it,...it->...t", x[..., ::skip], x[..., ::skip]))
        / net.N
    )

    ts = ts[::10]
    plt.plot(ts, autocorrs.mean(axis=0))
    plt.fill_between(
        ts,
        autocorrs.mean(axis=0) - autocorrs.std(axis=0),
        autocorrs.mean(axis=0) + autocorrs.std(axis=0),
        alpha=0.1,
        zorder=-1,
    )
    plt.ylim(0.0, net.Q0 * 1.3)

    plt.ylabel(r"$h^\alpha \cdot h^\alpha$")
    plt.xlabel(r"$t/\tau$")
    plt.axhline(net.Q0)
    ax = plt.gca()
    # add_tick(ax, net.Q0, "$Q_0$", which="y")
    # plt.tight_layout()
    if SHOW: show_plot()


class Multiple:
    def __init__(self, denominator=2, number=np.pi, latex="\pi"):
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


def tight_bbox(ax):
    fig = ax.get_figure()
    tight_bbox_raw = ax.get_tightbbox(fig.canvas.get_renderer())
    from matplotlib.transforms import TransformedBbox

    tight_bbox_fig = TransformedBbox(tight_bbox_raw, fig.transFigure.inverted())
    return tight_bbox_raw

LABEL_KWARGS = dict(size=12, weight='bold')
def add_panel_label(
        ax,
        letter,
        pad_x=0,
        pad_y=0,
        use_tight_bbox=False,
        ha="right",
        va="top",
        transform=None,
        return_text=False,
        x=0,
        y=1,
        **text_kwargs,
):

    if "$" in letter:
        letter_ = letter[1:-2]
    else:
        letter_ = letter

    letter = r"$\mathrm{\mathbf{" + letter_ + "}}$"

    if text_kwargs == {}:
        text_kwargs = LABEL_KWARGS

    if use_tight_bbox:
        bbox_fig = tight_bbox(ax)
        from matplotlib.transforms import TransformedBbox
        from matplotlib.transforms import Affine2D
        from matplotlib.transforms import Bbox

        fig = ax.get_figure()
        bbox_bare_fig = ax.get_position()

        w, h = fig.get_size_inches()
        bbox_bare_in = Bbox(
            [
                [bbox_bare_fig.y0 * w, bbox_bare_fig.y0 * h],
                [bbox_bare_fig.x1 * w, bbox_bare_fig.y1 * h],
            ]
        )
        bbox_ax = TransformedBbox(bbox_fig, ax.transAxes.inverted())
        bbox_tight_in = TransformedBbox(bbox_fig, Affine2D().scale(1.0 / fig.dpi))

        which_align = "left"
        # get diff in inches and convert to points
        start_x_pt = (
            abs(bbox_tight_in.xmin - bbox_bare_in.xmin) * 72
            if which_align == "left"
            else abs(bbox_tight_in.xmax - bbox_bare_in.xmax) * 72
        )

        text = ax.annotate(letter, xy=(0, 1), xytext=(-start_x_pt - pad_x, 0),
                           xycoords='axes fraction' if transform is None else transform, textcoords='offset points',
                           ha=ha, va=va, rotation=0, **LABEL_KWARGS)
    else:
        text = ax.set_title(
            letter,
            **text_kwargs,
            ha=ha,
            va="top",
            x=x - pad_x,
            y=y + pad_y,
            pad=0.0,
            transform=ax.transAxes if transform is None else transform,
        )

    return text

def format_angle(ax, n=2):
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / n))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter(n, np.pi)))

def align_labels(laxes):
    maxpad = 0
    for lax in laxes:
        maxpad = max(lax.yaxis.labelpad, maxpad)

    for lax in laxes:
        lax.yaxis.labelpad = maxpad


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

def get_limits(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if hasattr(ax, "get_zlim"):
        zlim = ax.get_zlim()
        return xlim, ylim, zlim
    else:
        return xlim, ylim


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

def data_lims_to_square_env(ax, margins=0.1):
    diffs = [np.diff(lim) for lim in get_limits(ax)]
    max_d = np.max(diffs)*(1+margins)
    cart = ["x", "y", "z"]
    for i, diff in enumerate(diffs):
        if diff < max_d:
            lim = getattr(ax, f"get_{cart[i]}lim")()
            pad_remain = max_d - diff
            getattr(ax, f"set_{cart[i]}lim")(
                lim[0] - pad_remain / 2, lim[1] + pad_remain / 2
            )

def save_plot(
        path,
        configs={},
        fn_dict={},
        anim=None,
        fig=None,
        file_formats=["png", "svg", "pdf"],
        use_hash=False,
        fn_prefix=None,
        include_filename=True,
        script_fn=None,
        **save_args
):
    config_ = {}
    for config in configs:
        try:
            config = asdict(config)
        except:
            pass
        config_ = config | config_

    try:
        path.mkdir()
    except:
        pass

    if use_hash:
        config_hash = joblib.hash(config_)[:4] if config_ != {} else ""
    else:
        config_hash = ""

    if script_fn is None:
       script_fn = Path(__file__).stem

    fn_ = (
            (script_fn if include_filename else "")
            + ("__" if fn_dict != {} else "")
            + "__".join([f"{k}_{v}" for k, v in fn_dict.items()])
            + ("__" + config_hash if config_hash else "")
    )
    if fn_prefix is not None:
        fn_prefix = fn_prefix + "_" + fn_
    else:
        fn_prefix = fn_

    if anim is None:
        # save_args.update(bbox_inches='tight', )
        for file_format in file_formats:
            if "png" in file_format: save_args.update({"dpi": 400})
            transparent = save_args.pop("transparent", True)
            if fig is None:
                plt.savefig(path / (fn_prefix + f".{file_format}"), transparent=transparent, **save_args)
            else:
                fig.savefig(path / (fn_prefix + f".{file_format}"), transparent=transparent, **save_args)
    else:
        plt.rcParams["animation.ffmpeg_path"] = "/usr/local/bin/ffmpeg"
        anim.save(path / (fn_prefix + ".mp4"), **save_args)

    def to_python_type(v):
        if np.isscalar(v):
            if np.issubdtype(type(v), np.integer):
                return int(v)
            elif np.issubdtype(type(v), np.floating):
                return float(v)
        else:
            return v
        
    with open(path / (fn_prefix + ".yml"), "w") as file:
        yaml.dump(
            {k: to_python_type(v) for k, v in config_.items() if (not hasattr(v, "__len__") or type(v) is str)},
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


def place_graphic(ax, inset_path, fit=None):
    fig = ax.get_figure()
    plt.rcParams['text.usetex'] = False
    # ax.axis("off")
    no_spine(ax, which="left", remove_all=True)
    no_spine(ax, which="bottom", remove_all=True)
    # no_spine(ax, which="right", remove_all=True)

    # freeze fig to finish off layout, new in 3.6
    fig.canvas.draw()
    fig.set_layout_engine(None)

    ax_bbox = ax.get_position()
    fig_w, fig_h = fig.get_size_inches()

    plt.rcParams.update(
        {
            "pgf.texsystem": "lualatex",
            "pgf.preamble": r"\usepackage{graphicx}\usepackage[export]{adjustbox}",
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
        path_str_pdf = str(temp_dir / (inset_path.stem + ".pdf"))
        p = subprocess.run(
            f"inkscape {path_str} --export-filename={path_str_pdf}", shell=True
        )
        path_str = path_str_pdf
    if fit is None:
        w, h = get_w_h(inset_path)
        if w / h > bbox["width"] / bbox["height"]:
            fit = "width"
        else:
            fit = "height"
    else:
        assert "width" in fit or "height" in fit

    tex_cmd = ""
    tex_cmd += r"\centering"
    tex_cmd += rf"\includegraphics[{fit}={{{bbox[fit]:.5f}in}}]{{{path_str}}}"
    print(bbox[fit])
    ax.text(0.0, 0.0, tex_cmd)


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
            line_args["markersize"] = 4
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


def multiple_formatter(denominator=2, number=np.pi, latex="\pi"):
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
    def __init__(self, denominator=2, number=np.pi, latex="\pi"):
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
        label="$x\sim Q_0/g^2$",
        color="C0",
        ls="dashed",
        lw=3,
    )
    ax.axhline(
        net.Q0,
        label="$h\sim Q_0$",
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


def rescale_rcparams(s):
    """
    From https://matplotlib.org/stable/users/explain/customizing.html
    https://seaborn.pydata.org/generated/seaborn.plotting_context.html#seaborn.plotting_context
    """
    import seaborn as sns
    ctx_dict = sns.plotting_context()

    update_dict = {}
    for k, v in ctx_dict.items():
        try:
            update_dict[k] = s * plt.rcParams[k] if s != -1 else plt.rcParamsDefault[k]
        except TypeError:
            print(f"Could not rescale {k}")

    plt.rcParams.update(update_dict)
    return plt



def plot_lmbds(ax, ranks, eigenvalues, normalize=False, cmap=None, line=False, scale=True, plot_sep=False, **largs):
    ranks = np.array(ranks)
    eigenvalues = np.array(eigenvalues)

    if normalize:
        eigenvalues /= eigenvalues[0]

    ranks = ranks[eigenvalues > 0]
    eigenvalues = eigenvalues[eigenvalues > 0]

    if ranks.size == 0:
        return

    marker = largs.pop("marker", "o")
    label = largs.pop("label", None)
    alpha = largs.pop("alpha", None)


    if plot_sep:
        for il, lmbd in enumerate(eigenvalues):
            if not "color" in largs and not "c" in largs and cmap is not None:
                largs["color"] = cmap(np.log10(lmbd) / np.log10(eigenvalues.max()))
            l_spec, = ax.plot(ranks[il], lmbd, marker=marker, ls='none',  alpha=largs.get("alpha", 0.4/np.max([1, 10**np.log10(ranks[il])/10])) if not alpha else alpha, label=label if il == 0 else None, **largs)

    else:
        l_spec, = ax.plot(ranks, eigenvalues, marker=marker, alpha=alpha, lw=plt.rcParams["lines.linewidth"]/2, **largs, zorder=-10)
    
    if line:
        l_spec, = ax.plot(ranks, eigenvalues, marker='none', alpha=0.8, lw=plt.rcParams["lines.linewidth"]/2, **largs, zorder=-10)

    # Make the plot double-logarithmic
    # check if a manual scale has been set
    if scale:
        if ax.get_yscale() == 'linear':
            if eigenvalues.min() > 0:
                ax.set_yscale('log')
            else:
                ax.set_yscale('symlog', linthresh=eigenvalues[eigenvalues > 0].min())
        ax.set_xscale('log') 

        ax.dataLim.x0 = 1
        ax.dataLim.x1 = ranks.max()

    return l_spec

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
