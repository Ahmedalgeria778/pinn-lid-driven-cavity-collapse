# -*- coding: utf-8 -*-
"""
Shared white / black-and-white print-friendly figure style for all publication
figures (AE request):

  * white background, black text/lines (no dark theme)
  * every variable carries its units, "(-)" when dimensionless
  * larger axis text (labels fs=14, ticks fs=12, suptitle fs=16)
  * curves distinguished by line style AND markers (square, star, triangle,
    diamond, ...) so that grayscale/black-and-white printing stays readable
  * grayscale fills + hatching for bars and fields
"""

import time
import os

import matplotlib.pyplot as plt

FACE = '#ffffff'      # figure / axes background
INK = '#000000'       # primary lines / text
GRAY = '#555555'      # secondary lines / annotations
GRID = '#9e9e9e'      # grid color

FS_LABEL = 14   # axis labels
FS_TICK = 12    # tick labels
FS_TITLE = 16   # figure suptitle
FS_AX_TITLE = 14

# --- per-profile / per-control line style + marker (B&W readable) -----------
PROFILE_LS = {'uniform': '-', 'sin_pi': '--', 'sin_2pi': '-.', 'pinn': ':'}
PROFILE_MK = {'uniform': 'o', 'sin_pi': 's', 'sin_2pi': '^', 'pinn': '*'}
TEMP_MK = {'pinn_t': 'D', 'cheb_t': 'X'}
TEMP_LS = {'pinn_t': '-', 'cheb_t': '--'}
RE_MK = {100: 'o', 500: 's', 1000: '^'}
RE_LS = {100: '-', 500: '--', 1000: '-.'}

# --- grayscale + hatch for grouped bars -------------------------------------
PROFILE_HATCH = {'uniform': '', 'sin_pi': '///', 'sin_2pi': '...', 'pinn': 'xxx'}
PROFILE_FILL = {'uniform': '#ffffff', 'sin_pi': '#c8c8c8', 'sin_2pi': '#8a8a8a',
                'pinn': '#e6e6e6'}
TEMP_HATCH = {'pinn_t': '///', 'cheb_t': 'xxx'}
TEMP_FILL = {'pinn_t': '#ffffff', 'cheb_t': '#bdbdbd'}


def style_axes(ax, xl='', yl='', title='', fs=FS_LABEL, tick=FS_TICK):
    """White-themed axes: black text/ticks, light dotted grid."""
    ax.set_facecolor(FACE)
    ax.tick_params(colors=INK, labelsize=tick)
    for sp in ax.spines.values():
        sp.set_edgecolor(INK)
    ax.grid(True, ls=':', lw=0.6, alpha=0.4, color=GRID)
    if xl:
        ax.set_xlabel(xl, color=INK, fontsize=fs)
    if yl:
        ax.set_ylabel(yl, color=INK, fontsize=fs)
    if title:
        ax.set_title(title, color=INK, fontsize=fs, fontweight='bold')


def figure_title(fig, txt, fs=FS_TITLE):
    fig.suptitle(txt, color=INK, fontsize=fs, fontweight='bold')


def legend(ax, fs=11, ncol=1, loc='best'):
    ax.legend(facecolor='white', edgecolor='#7a7a7a', labelcolor=INK,
              fontsize=fs, ncol=ncol, loc=loc)


def save_fig(fig, path, dpi=300):
    """Atomically save as PNG (temp file + replace) to survive OneDrive locks."""
    tmp = path + '.tmp'
    fig.savefig(tmp, dpi=dpi, format='png', bbox_inches='tight', facecolor=FACE, edgecolor=FACE)
    plt.close(fig)
    for _ in range(5):
        try:
            os.replace(tmp, path)
            return
        except OSError:
            time.sleep(1.0)
    os.replace(tmp, path)


def style_colorbar(cb, label='', fs=12):
    cb.ax.yaxis.set_tick_params(color=INK, labelcolor=INK, labelsize=11)
    if label:
        cb.set_label(label, color=INK, fontsize=fs)


def field_cmap():
    """Sequential grayscale colormap for field plots (B&W print)."""
    return plt.get_cmap('Greys')


def phase_cmap(n):
    """Monotone black->light-gray line colors for phase sequences."""
    return [('#%02x%02x%02x' % (int(255 * (1 - k / max(n - 1, 1))),
                                int(255 * (1 - k / max(n - 1, 1))),
                                int(255 * (1 - k / max(n - 1, 1)))))
            for k in range(n)]