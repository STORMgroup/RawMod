"""Can Dorado's basecall errors alone tell the native phage library from its PCR copy?

Per read: substitution / deletion / insertion rate at reference T, and at reference
positions with no T within 2 bp. Each read is scored on its own strand, so "T" means
a thymine of the sequenced strand (5hmU in the native library).

Reads are labelled from reads.tsv, so native and PCR reads come from ONE alignment
of ONE region. Read length is matched by decile before every AUROC, because the two
libraries have different length distributions.

Controls: runctrl = native run 1 vs native run 2, pcrctrl = PCR bc01 vs PCR bc05.
Both should give 0.5.
"""
import os as _os
from pathlib import Path as _Path
REPO = str(_Path(__file__).resolve().parents[2])
RAWMOD_RESULTS = _os.environ.get('RAWMOD_RESULTS') or f'{REPO}/paper/data/runs'  # training runs; shipped metrics by default
RAWMOD_DATA = _os.environ.get('RAWMOD_DATA', 'data')  # datasets (not shipped: set this to score raw signal)
RAWMOD_WORK = _os.environ.get('RAWMOD_WORK') or f'{REPO}/paper/data/work'  # outputs of the paper scripts; shipped by default
import sys, json, collections
import numpy as np, pysam, pandas as pd

ROOT = _os.environ.get('SPO1_DIR') or f'{RAWMOD_DATA}/SPO1'   # one directory per barcode: reads.tsv (read_id, group), windows.tsv, basecalled/reads_refined.sorted.bam
REF = _os.environ.get('SPO1_REF') or f'{ROOT}/SPO1_FJ230960.1.fasta'
COMP = str.maketrans('ACGT', 'TGCA')
NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
OUT = sys.argv[2] if len(sys.argv) > 2 else f'{RAWMOD_WORK}/basecall_errors/bc_strain_typing.json'
_os.makedirs(_os.path.dirname(OUT), exist_ok=True)
PROF = 10                                   # profile window around the nearest T

fa = pysam.FastaFile(REF); G = fa.fetch(fa.references[0]).upper()


def read_rows(ds):
    meta = pd.read_csv(f'{ROOT}/{ds}/reads.tsv', sep='\t')
    grp = dict(zip(meta.read_id, meta.group))
    bam = pysam.AlignmentFile(f'{ROOT}/{ds}/basecalled/reads_refined.sorted.bam')
    per = collections.Counter()
    rows, prof = [], collections.defaultdict(lambda: np.zeros((2 * PROF + 1, 2)))
    for r in bam.fetch(until_eof=True):
        if r.is_unmapped or r.is_secondary or r.is_supplementary:
            continue
        g = grp.get(r.query_name)
        if g is None or per[g] >= NMAX:
            continue
        per[g] += 1
        seq = r.query_sequence.upper(); rev = r.is_reverse
        qa, qe = r.query_alignment_start, r.query_alignment_end
        c = collections.defaultdict(lambda: np.zeros(4))      # cls -> n, sub, del, ins
        m = mm = d = ins = 0
        for qp, rp in r.get_aligned_pairs():
            if rp is None:
                if qa <= qp < qe:
                    ins += 1
                continue
            if rp < PROF + 2 or rp > len(G) - PROF - 3:
                continue
            rb = G[rp]
            rbs = rb.translate(COMP) if rev else rb
            ctx = G[rp - 2:rp + 3]
            ctxs = ctx.translate(COMP)[::-1] if rev else ctx
            cls = 'T' if rbs == 'T' else ('far' if 'T' not in ctxs else 'near')
            v = c[cls]; v[0] += 1
            err = 0
            if qp is None:
                v[2] += 1; d += 1; err = 1
            elif seq[qp] != rb:
                v[1] += 1; mm += 1; err = 1
            else:
                m += 1
            # distance to the nearest T on the read's own strand, signed
            win = G[rp - PROF:rp + PROF + 1]
            wins = win.translate(COMP)[::-1] if rev else win
            off = [i - PROF for i, b in enumerate(wins) if b == 'T']
            if off:
                k = min(off, key=abs)
                p = prof[g]; p[k + PROF, 0] += 1; p[k + PROF, 1] += err
        if c['T'][0] < 200 or c['far'][0] < 200:
            continue
        rows.append(dict(read=r.query_name, group=g, length=r.query_alignment_length,
                         qs=float(r.get_tag('qs')) if r.has_tag('qs') else np.nan,
                         identity=m / max(m + mm + d + ins, 1),
                         subT=c['T'][1] / c['T'][0], delT=c['T'][2] / c['T'][0],
                         subFar=c['far'][1] / c['far'][0], delFar=c['far'][2] / c['far'][0]))
    df = pd.DataFrame(rows)
    df['subT_minus_far'] = df.subT - df.subFar
    df['delT_minus_far'] = df.delT - df.delFar
    return df, {k: v.tolist() for k, v in prof.items()}


def auroc(pos, neg):
    x = np.concatenate([pos, neg]); y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    ok = np.isfinite(x); x, y = x[ok], y[ok]
    if y.sum() == 0 or (1 - y).sum() == 0:
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(x)
    return (r[y == 1].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (1 - y).sum())


def match_len(df, a, b, rng=np.random.default_rng(0)):
    """Subsample both groups to equal counts inside each decile of pooled read length."""
    e = np.quantile(df.length, np.linspace(0, 1, 11)); e[0] -= 1; e[-1] += 1
    keep = []
    for lo, hi in zip(e[:-1], e[1:]):
        s = df[(df.length > lo) & (df.length <= hi)]
        ia, ib = s.index[s.group == a], s.index[s.group == b]
        n = min(len(ia), len(ib))
        if n:
            keep += list(rng.choice(ia, n, replace=False)) + list(rng.choice(ib, n, replace=False))
    return df.loc[keep]


FEATS = ['identity', 'qs', 'subT', 'delT', 'subFar', 'delFar', 'subT_minus_far', 'delT_minus_far']
out = {}
for ds, (a, b) in [('bc01', ('mod_native', 'unmod_pcr')), ('bc02', ('mod_native', 'unmod_pcr')),
                   ('bc03', ('mod_native', 'unmod_pcr')), ('bc04', ('mod_native', 'unmod_pcr')),
                   ('bc05', ('mod_native', 'unmod_pcr')),
                   ('runctrl', ('mod_run1', 'mod_run2')), ('pcrctrl', ('pcr_bc01', 'pcr_bc05'))]:
    df, prof = read_rows(ds)
    if df.empty or not {a, b} <= set(df.group):
        print(ds, 'skipped', flush=True); continue
    mt = match_len(df, a, b)
    res = {'n_scored': {g: int((df.group == g).sum()) for g in (a, b)},
           'n_matched': {g: int((mt.group == g).sum()) for g in (a, b)},
           'median_length': {g: float(df.length[df.group == g].median()) for g in (a, b)},
           'median': {f: {g: float(df[f][df.group == g].median()) for g in (a, b)} for f in FEATS},
           'auroc_matched': {f: float(auroc(mt[f][mt.group == a].values, mt[f][mt.group == b].values)) for f in FEATS},
           'auroc_raw': {f: float(auroc(df[f][df.group == a].values, df[f][df.group == b].values)) for f in FEATS},
           'profile': prof}
    out[ds] = res
    print(ds, {f: round(res['auroc_matched'][f], 3) for f in FEATS}, flush=True)
    df.to_csv(OUT.replace('.json', f'_{ds}_reads.tsv.gz'), sep='\t', index=False)
open(OUT, 'w').write(json.dumps(out, indent=1))
print('wrote', OUT)
