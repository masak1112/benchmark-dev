"""Slide style for the S41 v1.5 helper-B figures (constants copied from ``s2storm/sci/slide_style.py`` on origin/aurora-port, not
imported across branches; model colours fixed by HELPERS.md 2026-11-12).

White, 16:9 (13.33 x 7.5 in), Open Sans / DejaVu, base 16 pt, no green, no figure-level title, small grey footer, grades on the figure.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

R = Path('/lus/flare/projects/S2Storm/s2storm')
V15 = R / 'runs/bing/eval/v1.5'
SLIDES = V15 / 'slides'
EXTRA = V15 / 'helperB' / 'figs'
W, H = 13.33, 7.5
MODEL = {'UC-S2S SST-anomaly': '#D55E00', 'UC-S2S plain': '#0072B2', 'FuXi-S2S': '#000000', 'IFS': '#CC79A7', 'ERA5': '#7a7a7a'}
SYS = {'ucs2s_sstanom': 'UC-S2S SST-anomaly', 'ucs2s_plain': 'UC-S2S plain', 'fuxi': 'FuXi-S2S'}
SYSC = {k: MODEL[v] for k, v in SYS.items()}
GREY, LGREY, SKY, ORANGE = '#7a7a7a', '#bdbdbd', '#56B4E9', '#E69F00'
DIV = 'RdBu_r'
REG = {'global': 'globe', 'tropics': 'tropics (20°S–20°N)', 'nh_extratropics': 'NH extratropics', 'sh_extratropics': 'SH extratropics',
       'conus': 'CONUS', 'land': 'land', 'ocean': 'ocean'}


def plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as p
    from cycler import cycler
    p.rcParams.update({'font.family': ['Open Sans', 'DejaVu Sans'], 'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16,
                       'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14, 'figure.facecolor': 'white',
                       'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False,
                       'axes.prop_cycle': cycler(color=[MODEL['UC-S2S SST-anomaly'], MODEL['UC-S2S plain'], MODEL['FuXi-S2S'], GREY, SKY])})
    return p


def fig(nrows=1, ncols=1, **kw):
    p = plt()
    kw.setdefault('figsize', (W, H))
    return p.subplots(nrows, ncols, **kw)


def footer(f, text, y=0.01):
    f.text(0.01, y, text, fontsize=11, color=GREY, ha='left', va='bottom', wrap=True)


def save(f, fid, name, footer_text=None, slide=True):
    """16:9 fixed canvas, < 1 MB (lower dpi if needed) -> path (slides/ or helperB/figs/)."""
    import matplotlib.pyplot as p
    out = SLIDES if slide else EXTRA
    out.mkdir(parents=True, exist_ok=True)
    if footer_text:
        footer(f, footer_text)
    path = out / f'{fid}_{name}.png'
    for dpi in (150, 130, 110, 90):
        f.savefig(path, dpi=dpi)
        if path.stat().st_size < 1_000_000:
            break
    p.close(f)
    return path


def caption(fid, line, slide=True):
    d = (SLIDES if slide else EXTRA) / 'captions'
    d.mkdir(parents=True, exist_ok=True)
    (d / f'{fid}.md').write_text(line.strip() + '\n')


def lsm15():
    return np.load(R / 'data/era5_bing_1p5/lsm.npy')


def coast(ax, lat_desc=False, lw=0.6):
    """coastlines from the ERA5 land-sea mask (no Natural Earth data on Aurora)."""
    m = lsm15()
    lat = np.linspace(-90, 90, 121)
    lon = np.arange(240) * 1.5
    ax.contour(lon, lat, m, levels=[0.5], colors='k', linewidths=lw)


def latlab(y):
    return 'EQ' if y == 0 else f'{int(y)}°N' if y > 0 else f'{int(-y)}°S'
