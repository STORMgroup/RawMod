"""One RawMod column scored on several basecalls of the same reads: per-row AUROC side by side.

    python -m bench.compare_inputs --inputs own1,hac1,fast1 [--subtool rawmod] [--holdout region]

Each inputs tag names a basecall of the benchmark read sets (BENCH_INPUTS; see README, "another basecalling model").
A tag may carry its own column name as tag:subtool when that basecall was scored under another registered name.
For every tag the rows are scored with bench.score under the given holdout, and the AUROCs are joined into one table,
written to $BENCH_WORK/compare_inputs_<subtool>_<tags>[_holdout-<name>].tsv and printed.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--inputs', default='own1,hac1,fast1')
    ap.add_argument('--subtool', default='rawmod')
    ap.add_argument('--holdout', default=os.environ.get('BENCH_HOLDOUT', 'region'))
    a = ap.parse_args()
    # spec: tag[/noref][:subtool]; /noref scores the unrefined-mapping run (BENCH_REFINE=none)
    specs = []
    for t in a.inputs.split(','):
        tag, _, subtool = t.partition(':'); subtool = subtool or a.subtool
        refine = 'none' if tag.endswith('/noref') else 'remora'; tag = tag.removesuffix('/noref')
        specs.append((tag, subtool, refine))
    tags = [tag + ('-noref' if refine == 'none' else '') for tag, _, refine in specs]
    frames = []
    for (tag, subtool, refine), label in zip(specs, tags):
        env = dict(os.environ, BENCH_INPUTS=tag, BENCH_HOLDOUT=a.holdout, BENCH_REFINE=refine)
        r = subprocess.run([sys.executable, '-m', 'bench.score', '--subtools', subtool], env=env, cwd=HERE,
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit(f'{tag}: bench.score failed\n{r.stderr[-2000:]}')
        from bench import common as C
        long = C.WORK / f"matrix_long_{tag}_k{C.K}_seed{C.SEED}_q{C.MIN_QS:g}{'_noref' if refine == 'none' else ''}.tsv"
        d = pd.read_csv(long, sep='\t'); tag = label
        d = d[d.subtool == subtool][['row', 'chem', 'status', 'n_sites', 'n_pos', 'frac_called', 'auroc', 'auroc_called']]
        frames.append(d.rename(columns={'auroc': tag, 'auroc_called': f'{tag}_called', 'frac_called': f'{tag}_frac', 'status': f'status_{tag}'}).set_index(['row', 'chem']))
    base = frames[0][['n_sites', 'n_pos']]
    t = base.join([f[[tag, f'{tag}_called', f'{tag}_frac', f'status_{tag}']] for f, tag in zip(frames, tags)])
    t = t.reset_index()
    summary = {tag: t[tag].mean() for tag in tags}
    per_mod = {tag: t.groupby('chem')[tag].mean().mean() for tag in tags}
    called = {tag: t[f'{tag}_called'].mean() for tag in tags}
    from bench import common as C
    out = C.WORK / f"compare_inputs_{a.subtool}_{'_'.join(tags)}{'_holdout-' + a.holdout if a.holdout else ''}.tsv"
    t.to_csv(out, sep='\t', index=False, float_format='%.4f')
    with pd.option_context('display.width', 200, 'display.max_rows', 100):
        cols = ['row', 'chem', 'n_sites', 'n_pos'] + [c for tag in tags for c in (tag, f'{tag}_called', f'{tag}_frac')]
        print(t[cols].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print('\nmean over rows:          ' + '  '.join(f'{k} {v:.3f}' for k, v in summary.items()))
    print('mean over modifications: ' + '  '.join(f'{k} {v:.3f}' for k, v in per_mod.items()))
    print('mean over rows, called sites only: ' + '  '.join(f'{k} {v:.3f}' for k, v in called.items()))
    print('wrote', out)


if __name__ == '__main__':
    main()
