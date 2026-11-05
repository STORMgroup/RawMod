"""Table 1 with 95% confidence intervals (supp_bench_ci.tex, split into two panels), and the called-only / call-rate table.

  python -m bench.ci_table [ROWS.tsv] [OUT_DIR]

ROWS.tsv is written by `python -m bench.ci` (default: next to the other tables of 2_score.sh); the paper's rows are
shipped in paper/data/work/benchmark_ci/ci_rows.tsv and are used when the work tree has none. Tables go to OUT_DIR
(default: the tables directory of 2_score.sh)."""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
_REPO = Path(__file__).resolve().parents[3]
os.environ.setdefault('BENCH_HOLDOUT', 'region'); os.environ.setdefault('BENCH_WORK', str(_REPO / 'test' / 'work'))
from . import common as C
from .paper_table import MODELS, MAIN_DROP, GROUPS
from .upright import fit

_paper = C.WORK / f'paper_{C.PAPER_TAG}'
_rows = Path(sys.argv[1]) if len(sys.argv) > 1 else (_paper / 'ci_rows.tsv' if (_paper / 'ci_rows.tsv').exists() else _REPO / 'paper/data/work/benchmark_ci/ci_rows.tsv')
D = pd.read_csv(_rows, sep='\t').set_index('row')
cols = [(m, lab) for m, lab in MODELS if m not in MAIN_DROP]
LAB = {r: l for g, mem in GROUPS for r, l in mem}
HEAD = {g: g for g, _ in GROUPS}

def ci(r, m):
    a, lo, hi = D.loc[r, m], D.loc[r, m + '::lo'], D.loc[r, m + '::hi']
    if not np.isfinite(a): return '--'
    return r"\shortstack{%.3f\\\tiny{[%.2f,%.2f]}}" % (a, lo, hi)

def called(r, m):
    a, f = D.loc[r, m + '::called'], D.loc[r, m + '::frac_called']
    if not np.isfinite(f): return '--'
    v = '--' if not np.isfinite(a) else f"{a:.3f}"
    return r"\shortstack{%s\\\tiny{%.0f\%%}}" % (v, f * 100)

def table(cellfn, path, ncol_note):
    L = [r'\setlength{\tabcolsep}{2pt}%',   # the paper's \fittable shrinks the table to the line
         r'\begin{tabular}{@{}l' + 'c' * (len(cols) + 1) + r'@{}}', r'\toprule',
         r'\textbf{Dataset} & \textbf{Sites} & ' + ' & '.join(r'\textbf{' + l + '}' for _, l in cols) + r' \\',
         r'\midrule']
    for g, members in GROUPS:
        L.append(r'\multicolumn{%d}{@{}l}{\textbf{%s}} \\' % (len(cols) + 2, HEAD[g]))
        for row, lab in members:
            n = int(D.loc[row, 'n_sites'])
            L.append(r'\hspace{3pt}' + lab + f' & {n:,} & ' + ' & '.join(cellfn(row, m) for m, _ in cols) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}']
    open(path, 'w').write('\n'.join(L) + '\n')
    print('wrote', path)

out = Path(sys.argv[2]) if len(sys.argv) > 2 else _paper / 'tables'
out.mkdir(parents=True, exist_ok=True)
table(ci, str(out / 'supp_bench_ci.tex'), None)
for p in fit(out / 'supp_bench_ci.tex'):
    print('wrote', p)
table(called, str(out / 'supp_bench_called.tex'), None)
