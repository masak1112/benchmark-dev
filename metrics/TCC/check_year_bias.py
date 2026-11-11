"""Mean weeks 3-4 anomaly per init year (forecast vs ERA5) for the pooled multicond 2019+2020 run and V2:
a model's bias (forecast - ERA5) should be about the same in both years."""
import numpy as np
import tcc_weekly as t

t.N_MEMBERS = 20
dates = [l.strip() for l in open('dates_mc1920n.txt') if l.strip()]
wk = t.WEEK_NAMES.index('week3-4')
for model in ('ucs2s_multicond1920n', 'ucs2s_v2_mcpair2'):
    used, fa, oa = t.load_anomalies(model, dates)
    y = np.array([d[:4] for d in used])
    # same months in both years (Jan 3 - Feb 3) so the bias comparison is not a seasonal effect
    jan = np.array([d[4:] <= '0203' for d in used])
    for lbl in ('t2m', 'z500'):
        i = t.LABELS.index(lbl)
        for reg in ('ocean', 'land', 'global'):
            m = t.MASKS[reg]
            rm = lambda x: np.nansum(t.W2D[m] * x[..., m], axis=-1) / t.W2D[m].sum()
            f, o = rm(fa[:, i, wk]), rm(oa[:, i, wk])
            b = {yr: (f[(y == yr) & jan] - o[(y == yr) & jan]).mean() for yr in ('2019', '2020')}
            print(f'{lbl:5s}{model:22s}{reg:7s} bias (Jan 3-Feb 3) 2019 {b["2019"]:+8.2f}  2020 {b["2020"]:+8.2f}')
    del fa, oa
