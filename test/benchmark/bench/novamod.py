#!/usr/bin/env python3
"""NovaMod (Zou, Ahsan & Wang, Genes 2026): unsupervised modification detection with a variational autoencoder of
unmodified signal, run as a user without RawMod would run it, and scored per read like every other column.

NovaMod learns the signal of unmodified DNA (a CNN-Transformer VAE over 9-mer windows of per-base signal statistics
and base qualities) and scores a base of a read by its reconstruction error (score_recon): high = unlike unmodified DNA.
  - Two-sample rows (modified library vs. its unmodified control library): one model per control library, trained with
    the released recipe (training/configs/train.example.json: 9-mers, 20 epochs) on every primary read of that library
    with mean q-score >= 10 that is in no benchmark read set -- including its reads in the held-out regions (NOT held
    out: this is how a user would train it on their own control). Libraries as in bench.uncalled4_controls.SOURCES.
    Subtool 'novamod'; blank on one-sample rows (no control library). The checkpoints the README calls released
    (online_test9/online_test10) are not in the repository.
  - Subtool 'novamod_pooled': ONE model, same recipe, trained on all four control libraries' training reads pooled
    (oligonucleotide control, E. coli dam- dcm-, H. pylori 26695 WGA, SPO1 PCR; one SignalBAM over the concatenated
    references), scored on every row -- the generalised NovaMod (no read of the human, plant, mouse or Anabaena
    genomes is in its training data).
Scoring: val.py in 'site' mode on the sample's benchmark read set at the benchmark's candidate sites; every read
covering a site on the site's strand gets its score_recon there, and bench.aggregate keeps the K selected reads
(per-site score = mean score_recon over them, as for every per-read tool).
Every read set is first turned into a SignalBAM with NovaMod's scripts/pod5_to_bam.py (raw signal in tag SG); our
Dorado 1.4.0 sup@v5.0.0 basecalls already carry the mv/ts/sm/sd/sv tags it needs.

  python -m bench.novamod --train spo1_bc01  # one control model
  python -m bench.novamod --train-whitelist spo1_bc01_regC  # control model of a whitelisted control read set (clean SPO1 row)
  python -m bench.novamod --sample spo1_bc07_reg # calls of one sample (or rawmod_bench.py task tool --subtool novamod)
"""
import argparse
import copy
import gzip
import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pysam

from . import common as C
from . import rawmod_cands
from . import settings as ST

RECIPE = 'configs/train.example.json'
EPOCHS = 20  # train.example.json; the last epoch is used
POOL = ('syn_control', 'ecoli_dm', 'hp26695_wga', 'spo1_bc01')
MIN_QS = 10


def repo():
    return Path(ST.need('NOVAMOD', 'NovaMod repository (github.com/WGLab/novamod)'))


def work():
    return Path(ST.need('NOVAMOD_WORK', 'NovaMod models and SignalBAMs'))


def py():
    return ST.need('NOVAMOD_PY', 'python of the NovaMod environment')


def ncpu():
    return len(os.sched_getaffinity(0))


def sh(cmd, **kw):
    print(cmd if isinstance(cmd, str) else ' '.join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, shell=isinstance(cmd, str), **kw)


def signal_bam(bam, pod5, out, name):
    """SignalBAM <out>/<name>.final.bam of an aligned BAM (dorado tags) and its POD5 (scripts/generate_features.sh)."""
    final = out / f'{name}.final.bam'
    if final.exists():
        return final
    tmp = out / f'{name}_chunks'
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    sh([py(), repo() / 'scripts' / 'pod5_to_bam.py', '--bam', bam, '--input', pod5, '--file_type', 'pod5',
        '--output', tmp, '--prefix', 'converted', '--threads', str(min(16, ncpu()))])
    sam = ST.get('SAMTOOLS', 'samtools')
    chunks = ' '.join(str(p) for p in sorted(tmp.glob('converted*bam')))
    sh(f'{sam} cat {chunks} -o - | {sam} sort -m 2G -O BAM -o {final}.tmp.bam --write-index -@ {min(8, ncpu())} - '
       f'&& mv {final}.tmp.bam {final} && mv {final}.tmp.bam.csi {final}.csi')
    shutil.rmtree(tmp)
    return final


def manifest(d, data_id, bam, ref):
    p = d / f'manifest_{data_id}.csv'
    pd.DataFrame([dict(data_id='ref', condition='ref', reference='', type='', size='', file_path=ref, label_path='', notes=''),
                  dict(data_id=data_id, condition='unmod', reference='ref', type='dna', size='full', file_path=str(bam),
                       label_path='', notes='')]).to_csv(p, index=False)
    return p


def train(lib):
    """Control model of one control library -> $NOVAMOD_WORK/<lib>/state_dicts/model_v1_cnn_t/novamod_<lib>-epoch20.pt"""
    from .uncalled4_controls import SOURCES
    d = work() / lib
    ck = d / 'state_dicts' / 'model_v1_cnn_t' / f'novamod_{lib}-epoch{EPOCHS}.pt'
    if ck.exists():
        print(f'{lib}: {ck} exists'); return ck
    d.mkdir(parents=True, exist_ok=True)
    bam, pod5, ref_sample, excl_sets = SOURCES[lib]
    if not (d / 'train.final.bam').exists():
        excl = set()
        for s in excl_sets:
            excl |= set(gzip.open(ST.DATA_DIR / 'read_ids' / f'{s}.txt.gz', 'rt').read().split())
        ids = d / 'train_ids.txt'
        n = 0
        with pysam.AlignmentFile(bam, check_sq=False) as f, pysam.AlignmentFile(str(d / 'train.unsorted.bam'), 'wb', template=f) as g:
            keep = []
            for r in f.fetch(until_eof=True):
                if (r.is_unmapped or r.is_secondary or r.is_supplementary or r.query_name in excl or not r.has_tag('mv')
                        or (r.get_tag('qs') if r.has_tag('qs') else 99) < MIN_QS):
                    continue
                g.write(r); keep.append(r.query_name); n += 1
        ids.write_text('\n'.join(keep) + '\n')
        pysam.sort('-@', '8', '-o', str(d / 'train.bam'), str(d / 'train.unsorted.bam')); os.remove(d / 'train.unsorted.bam')
        pysam.index(str(d / 'train.bam'))
        if not (d / 'train.pod5').exists():
            rec = '--recursive ' if Path(pod5).is_dir() else ''
            sh(f"{ST.get('POD5', 'pod5')} filter {pod5} {rec}--ids {ids} --missing-ok --output {d}/train.pod5.tmp "
               f"&& mv {d}/train.pod5.tmp {d}/train.pod5")
        print(f'{lib}: {n:,} training reads (primary, qs >= {MIN_QS}, in no benchmark read set of {", ".join(excl_sets)})')
        signal_bam(d / 'train.bam', d / 'train.pod5', d, 'train')
        for f in ('train.pod5', 'train.bam', 'train.bam.bai'):
            (d / f).unlink(missing_ok=True)
    cfg = json.loads((repo() / 'training' / RECIPE).read_text())
    cfg['run']['run_name'] = f'novamod_{lib}'
    cfg['data']['data_id'] = lib
    cfg['data']['manifest_path'] = str(manifest(d, lib, d / 'train.final.bam', C.samples()[ref_sample]['ref']))
    cfg['loader']['num_workers'] = max(1, ncpu() - 2)
    assert cfg['train']['epochs'] == EPOCHS
    (d / 'train.json').write_text(json.dumps(cfg, indent=2))
    sh([py(), repo() / 'training' / 'train.py', '--config', d / 'train.json'], cwd=d)
    if not ck.exists():
        raise SystemExit(f'{lib}: training ended without {ck}')
    return ck


def train_pooled():
    """One model on the four control libraries' training SignalBAMs merged (bench.novamod --train <lib> first)."""
    d = work() / 'pooled'
    ck = d / 'state_dicts' / 'model_v1_cnn_t' / f'novamod_pooled-epoch{EPOCHS}.pt'
    if ck.exists():
        print(f'pooled: {ck} exists'); return ck
    from .uncalled4_controls import SOURCES
    d.mkdir(parents=True, exist_ok=True)
    parts = [work() / lib / 'train.final.bam' for lib in POOL]
    miss = [str(p) for p in parts if not p.exists()]
    if miss:
        raise SystemExit(f'pooled: missing training SignalBAMs {miss}; run --train {" ".join(POOL)} first')
    ref = d / 'ref.fa'
    if not ref.exists():
        with open(f'{ref}.tmp', 'w') as fo:
            for lib in POOL:
                fo.write(Path(C.samples()[SOURCES[lib][2]]['ref']).read_text().rstrip('\n') + '\n')
        os.replace(f'{ref}.tmp', ref)
        pysam.faidx(str(ref))
    bam = d / 'train.final.bam'
    if not bam.exists():
        sam = ST.get('SAMTOOLS', 'samtools')
        sh(f"{sam} merge -f -@ {min(8, ncpu())} --write-index -o {bam}.tmp.bam {' '.join(map(str, parts))} "
           f"&& mv {bam}.tmp.bam {bam} && mv {bam}.tmp.bam.csi {bam}.csi")
    cfg = json.loads((repo() / 'training' / RECIPE).read_text())
    cfg['run']['run_name'] = 'novamod_pooled'
    cfg['data']['data_id'] = 'pooled'
    cfg['data']['manifest_path'] = str(manifest(d, 'pooled', bam, str(ref)))
    cfg['loader']['num_workers'] = max(1, ncpu() - 2)
    assert cfg['train']['epochs'] == EPOCHS
    (d / 'train.json').write_text(json.dumps(cfg, indent=2))
    sh([py(), repo() / 'training' / 'train.py', '--config', d / 'train.json'], cwd=d)
    if not ck.exists():
        raise SystemExit(f'pooled: training ended without {ck}')
    return ck


def model_for(sample, subtool):
    """(work dir holding state_dicts/ and train.json, run name, epoch, control) for a sample and subtool."""
    if subtool == 'novamod_pooled':
        return work() / 'pooled', 'novamod_pooled', EPOCHS, 'pooled'
    from .uncalled4_sites import control_lib, control_sample
    from .readsel import whitelist
    lib = control_lib(sample)
    if not lib:
        return None
    cs = control_sample(sample)
    if whitelist(cs) is not None:  # opt-in (config/read_whitelist.tsv): the control model trained on the whitelisted control reads
        return work() / cs, f'novamod_{cs}', EPOCHS, cs
    return work() / lib, f'novamod_{lib}', EPOCHS, lib


def train_whitelist(cs):
    """Control model of a whitelisted control read set (config/read_whitelist.tsv ): the same recipe as train(),
    on the sample's whitelisted reads (primary, qs >= MIN_QS, with move tables) -- for the clean SPO1 row, the bc01 reads
    anchored at a PCR primer / product end. -> $NOVAMOD_WORK/<sample>/state_dicts/model_v1_cnn_t/novamod_<sample>-epoch20.pt"""
    from .readsel import whitelist
    wl = whitelist(cs)
    if wl is None:
        raise SystemExit(f'{cs}: no entry in config/read_whitelist.tsv')
    S = C.samples()[cs]
    d = work() / cs
    ck = d / 'state_dicts' / 'model_v1_cnn_t' / f'novamod_{cs}-epoch{EPOCHS}.pt'
    if ck.exists():
        print(f'{cs}: {ck} exists'); return ck
    d.mkdir(parents=True, exist_ok=True)
    if not (d / 'train.final.bam').exists():
        ids = d / 'train_ids.txt'
        keep = []
        with pysam.AlignmentFile(S['sub_bam']) as f, pysam.AlignmentFile(str(d / 'train.unsorted.bam'), 'wb', template=f) as g:
            for r in f.fetch(until_eof=True):
                if (r.is_unmapped or r.is_secondary or r.is_supplementary or r.query_name not in wl or not r.has_tag('mv')
                        or (r.get_tag('qs') if r.has_tag('qs') else 99) < MIN_QS):
                    continue
                g.write(r); keep.append(r.query_name)
        ids.write_text('\n'.join(keep) + '\n')
        pysam.sort('-@', '8', '-o', str(d / 'train.bam'), str(d / 'train.unsorted.bam')); os.remove(d / 'train.unsorted.bam')
        pysam.index(str(d / 'train.bam'))
        if not (d / 'train.pod5').exists():
            sh(f"{ST.get('POD5', 'pod5')} filter {S['sub_pod5']} --ids {ids} --missing-ok --output {d}/train.pod5.tmp "
               f"&& mv {d}/train.pod5.tmp {d}/train.pod5")
        print(f'{cs}: {len(keep):,} training reads (whitelisted, primary, qs >= {MIN_QS})')
        signal_bam(d / 'train.bam', d / 'train.pod5', d, 'train')
        for f in ('train.pod5', 'train.bam', 'train.bam.bai'):
            (d / f).unlink(missing_ok=True)
    cfg = json.loads((repo() / 'training' / RECIPE).read_text())
    cfg['run']['run_name'] = f'novamod_{cs}'
    cfg['data']['data_id'] = cs
    cfg['data']['manifest_path'] = str(manifest(d, cs, d / 'train.final.bam', S['ref']))
    cfg['loader']['num_workers'] = max(1, ncpu() - 2)
    assert cfg['train']['epochs'] == EPOCHS
    (d / 'train.json').write_text(json.dumps(cfg, indent=2))
    sh([py(), repo() / 'training' / 'train.py', '--config', d / 'train.json'], cwd=d)
    if not ck.exists():
        raise SystemExit(f'{cs}: training ended without {ck}')
    return ck


def run(sample, subtool):
    """Per-read NovaMod scores at the sample's candidate sites -> $BENCH_WORK/<sample>/<subtool>/<TAG>/calls.tsv.gz"""
    S = C.samples()[sample]
    out = C.WORK / sample / subtool / C.TAG
    if (out / 'calls.tsv.gz').exists():
        print(f'{sample} {subtool}: done'); return
    out.mkdir(parents=True, exist_ok=True)
    m = model_for(sample, subtool)
    if m is None:
        print(f'{sample} {subtool}: no unmodified control library -- not scored (blank in the tables)'); return
    mdir, run_name, ep, lib = m
    ck = mdir / 'state_dicts' / 'model_v1_cnn_t' / f'{run_name}-epoch{ep}.pt'
    if not ck.exists():
        raise SystemExit(f'{sample}: NovaMod model {ck} missing; run python -m bench.novamod --train {lib}')
    sbam = signal_bam(S['sub_bam'], S['sub_pod5'], out, 'reads')
    rawmod_cands.build(sample, out=out)
    bed = out / 'sites.bed6'
    with open(bed, 'w') as fo:
        for name, st in (('plus', '+'), ('minus', '-')):
            b = out / f'cand_{name}.bed'
            if b.stat().st_size:
                for c, p in pd.read_csv(b, sep='\t', header=None, usecols=[0, 1], dtype={0: str}).itertuples(index=False):
                    fo.write(f'{c}\t{p}\t{p + 1}\t.\t0\t{st}\n')
    cfg = json.loads((repo() / 'training' / 'configs' / 'val.test10-hg004.json').read_text())  # released scoring recipe
    rcfg = json.loads((mdir / 'train.json').read_text())
    v = copy.deepcopy(cfg)
    v['run']['run_name'] = run_name
  # model and loss settings exactly as trained
    v['model'], v['data']['sampling']['kmer_len'] = rcfg['model'], rcfg['data']['sampling']['kmer_len']
    v['train'] = {k: rcfg['train'][k] for k in ('recon', 'huber_delta', 'nll_var', 'kmer_weight_center_base', 'kmer_weight_sd')}
    v['data']['manifest_path'] = str(manifest(out, 'sample', sbam, S['ref']))
    v['val'] = {'checkpoint_epoch': ep, 'output_dir': str(out),
                'loader': {'batch_size': 256, 'num_workers': max(1, min(16, ncpu() - 2))},
                'datasets': [{'name': 'sites', 'data_id': 'sample', 'method': 'site', 'spec': str(bed),
                              'max_lines': 10 ** 12, 'chromosomes': []}]}
    (out / 'val.json').write_text(json.dumps(v, indent=2))
    sh([py(), repo() / 'training' / 'val.py', '--config', out / 'val.json'], cwd=mdir)
    pq = out / f'{run_name}-sites.pq'
    df = pd.read_parquet(pq, columns=['read_name', 'chrom', 'ref_pos_0based', 'strand', 'score_recon'])
    df = df.rename(columns={'read_name': 'read_id', 'chrom': 'contig', 'ref_pos_0based': 'pos', 'score_recon': 'prob'})
    df[['read_id', 'contig', 'pos', 'strand', 'prob']].to_csv(out / 'calls.tsv.gz.tmp', sep='\t', index=False,
                                                              float_format='%.6g', compression='gzip')
    os.replace(out / 'calls.tsv.gz.tmp', out / 'calls.tsv.gz')
    pq.unlink()
    (out / 'PROVENANCE.txt').write_text(f'NovaMod {repo()} val.py site mode, checkpoint {ck} (control: {lib}; '
                                        f'not held out); prob = score_recon\n')
    print(f'{sample} {subtool}: {len(df):,} per-read scores at {df.groupby(["contig", "pos", "strand"]).ngroups:,} sites')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train', nargs='*', default=[], help="control libraries, or 'pooled'")
    ap.add_argument('--train-whitelist', nargs='*', default=[], help='whitelisted control read sets (config/read_whitelist.tsv)')
    ap.add_argument('--sample', nargs='*', default=[])
    ap.add_argument('--subtool', default='novamod')
    a = ap.parse_args()
    for lib in a.train:
        train_pooled() if lib == 'pooled' else train(lib)
    for cs in a.train_whitelist:
        train_whitelist(cs)
    for s in a.sample:
        run(s, a.subtool)


if __name__ == '__main__':
    main()
