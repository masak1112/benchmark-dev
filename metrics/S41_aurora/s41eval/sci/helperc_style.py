"""Slide style for the S41 helper-C figures (2026-11-13 set). Constants copied (not imported) from ``s2storm/sci/slide_style.py`` on
branch ``aurora-port``, with the model colours fixed by v1.5/HELPERS.md for this set. White, 16:9, Open Sans / DejaVu, base 16 pt, no
green, no figure-level title, small grey footer, grades written on the figure."""
from __future__ import annotations

from pathlib import Path

R = Path('/lus/flare/projects/S2Storm/s2storm')
V15 = R / 'runs' / 'bing' / 'eval' / 'v1.5'
SLIDES = V15 / 'slides'
EXTRA = V15 / 'helperC' / 'figs'
W, H = 13.33, 7.5
MODEL = {'UC-S2S SST-anomaly': '#D55E00', 'UC-S2S plain': '#0072B2', 'FuXi-S2S': '#000000', 'IFS': '#CC79A7', 'ERA5': '#7a7a7a'}
SYS = {'ucs2s_sstanom': 'UC-S2S SST-anomaly', 'ucs2s_plain': 'UC-S2S plain', 'fuxi': 'FuXi-S2S', 'ifs': 'IFS'}
GREY, LGREY, SKY, ORANGE = '#7a7a7a', '#bdbdbd', '#56B4E9', '#E69F00'
DIV = 'RdBu_r'
PROVISIONAL_MJO = 'MJO phases: EOF pair rotated 37.5° to BoM (fixed, same for all systems) — provisional'


def plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as p
    p.rcParams.update({'font.family': ['Open Sans', 'DejaVu Sans'], 'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16,
                       'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14, 'figure.facecolor': 'white',
                       'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False})
    return p


def fig(nrows=1, ncols=1, **kw):
    p = plt()
    kw.setdefault('figsize', (W, H))
    return p.subplots(nrows, ncols, **kw)


def footer(f, text, y=0.01, width=178):
    import textwrap
    lines = sum((textwrap.wrap(t, width) or [''] for t in text.split('\n')), [])
    f.text(0.01, y, '\n'.join(lines), fontsize=11, color=GREY, ha='left', va='bottom', linespacing=1.15)


def save(f, fid, name, footer_text=None, slide=True):
    """16:9 fixed canvas, < 1 MB; slide=True -> v1.5/slides/, else v1.5/helperC/figs/."""
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


def key_numbers(fid, lines):
    """replace the block of figure ``fid`` in slides/key_numbers/helperC.md (lines: list of 'number · file/row')."""
    d = SLIDES / 'key_numbers'
    d.mkdir(parents=True, exist_ok=True)
    f = d / 'helperC.md'
    txt = f.read_text() if f.exists() else '# Key numbers, helper C (S41 W5/W5b/W6; all exploratory — none is a pre-registered row). Files are under v1.5/helperC/: C01 mjo/, C02 hov/, C03 waves/, C04 tele/, C05 strat/, C06 enso/, C07 windows/ and c07_era5/. Brackets = 95 % block-bootstrap CI.\n'
    blocks, cur = {}, None
    head = []
    for ln in txt.splitlines():
        if ln.startswith('## '):
            cur = ln[3:].split()[0]
            blocks[cur] = [ln]
        elif cur is None:
            head.append(ln)
        else:
            blocks[cur].append(ln)
    blocks[fid] = [f'## {fid}'] + [f'- {x}' for x in lines] + ['']
    out = head + [''] + sum((blocks[k] for k in sorted(blocks)), [])
    f.write_text('\n'.join(out).replace('\n\n\n', '\n\n'))
