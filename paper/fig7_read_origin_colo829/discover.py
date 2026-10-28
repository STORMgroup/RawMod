#!/usr/bin/env python3
"""Label-free read partition per window: at every site the reads are split in two by the sign of the first principal component of their embeddings; a co-association consensus over the sites of the window gives one partition and one continuous score per read. Labels enter only to report ARI and oriented AUROC against the true read origin."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

SEED = 0


def site_split(X: np.ndarray, seed: int = SEED,
               method: str = 'gmm') -> np.ndarray | None:
    """Label-free 2-way split of the reads at one site. Returns +-1 per read.

    method 'gmm' is the original; 'pc1' takes the sign of the first principal
    component. ablate_site_split.py found them equivalent (dose-response rho
    0.742 vs 0.768, medians within noise at n=24) because the cross-site
    consensus supplies the robustness, not the per-site model. 'pc1' is
    deterministic — so seed bagging is a no-op — and about 10x faster, which
    matters once a window carries ~260 sites at ~100 reads each.
    """
    if len(X) < 8:
        return None
    Xs = StandardScaler().fit_transform(X)
    npc = int(min(10, max(2, len(X) // 4), Xs.shape[0] - 1, Xs.shape[1]))
    Xp = PCA(n_components=npc, random_state=seed).fit_transform(Xs).astype(np.float64)
    if method == 'pc1':
        z = Xp[:, 0]
        return np.where(z >= 0, 1.0, -1.0) if z.std() > 1e-12 else None
    try:
        lab = GaussianMixture(2, random_state=seed, n_init=5,
                              covariance_type='diag',
                              reg_covar=1e-4).fit_predict(Xp)
    except Exception:
        from sklearn.cluster import KMeans
        lab = KMeans(2, random_state=seed, n_init=10).fit_predict(Xp)
    if len(np.unique(lab)) < 2:
        return None
    return np.where(lab == 0, -1.0, 1.0)


def consensus(splits: dict, reads: list) -> tuple:
    """Co-association -> leading eigenvector -> (read scores, site loadings).

    S is summed over sites; N counts how many sites each pair actually shared,
    so pairs that co-occur rarely are not inflated. The row/column mean is
    removed before the eigendecomposition, otherwise the leading eigenvector is
    just the all-ones vector (every pair agrees at homogeneous sites, which
    carries no partition information).
    """
    idx = {r: i for i, r in enumerate(reads)}
    n = len(reads)
    S = np.zeros((n, n))
    N = np.zeros((n, n))
    for si, (rids, s) in splits.items():
        j = np.array([idx[r] for r in rids])
        S[np.ix_(j, j)] += np.outer(s, s)
        N[np.ix_(j, j)] += 1.0
    with np.errstate(invalid='ignore', divide='ignore'):
        C = np.where(N > 0, S / np.maximum(N, 1), 0.0)
    np.fill_diagonal(C, 0.0)
    C = C - C.mean(axis=0, keepdims=True) - C.mean(axis=1, keepdims=True) + C.mean()
    w, v = np.linalg.eigh(C)
    score = v[:, -1] * np.sign(np.sum(v[:, -1]) or 1.0)
    part = np.where(score >= 0, 1, 0)
    load = {}
    for si, (rids, s) in splits.items():
        j = np.array([idx[r] for r in rids])
        z = score[j]
        if np.std(z) < 1e-12:
            load[si] = 0.0
        else:
            load[si] = float(abs(np.corrcoef(s, z)[0, 1]))
    return score, part, load, float(w[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True, help='from emb_all_reads.py')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--per-window', action='store_true',
                    help='run the consensus inside each window separately')
    ap.add_argument('--truth-sites', default=None,
                    help='sites.tsv — evaluation only, for site precision/recall')
    ap.add_argument('--haplotype', default=None,
                    help='TSV read_id<TAB>window<TAB>hap — evaluation only')
    ap.add_argument('--read-meta', default=None,
                    help='reads.tsv, for the flowcell batch check — eval only')
    ap.add_argument('--min-reads', type=int, default=12)
    ap.add_argument('--pos-group', default='tumor',
                    help='which group label counts as the positive class when '
                         'scoring; falls back to the lexicographically second '
                         'group present, so non-COLO829 pairings work unchanged')
    ap.add_argument('--split-method', default='gmm', choices=['gmm', 'pc1'],
                    help='per-site 2-way split; see site_split')
    ap.add_argument('--emit-reads', action='store_true',
                    help='also write discovered_reads.tsv: one row per '
                         '(window, read) with the consensus score and side. '
                         'The summary is per-window only, which makes any '
                         'read-level grounding — orienting the split by bulk '
                         'methylation, or checking it against SNV-carrying '
                         'reads — impossible after the fact.')
    ap.add_argument('--n-seeds', type=int, default=5,
                    help='bag the per-site GMM over this many seeds. A single '
                         'seed is unstable where the signal is weak: on the '
                         'pilot, one window swung between ARI 0.01 and 0.68 '
                         'across seeds. Bagging averages the co-association '
                         'over seeds before the eigendecomposition.')
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(a.npz, allow_pickle=True)
    emb, rid, si = d['emb'], d['read_id'], d['site_i']
    grp, win = d['group'], d['window']
    s_pos, s_win = d['site_pos'], d['site_window']

    truth = None
    if a.truth_sites:
        t = pd.read_csv(a.truth_sites, sep='\t')
        truth = {(r.chrom, int(r.start)): r.cls for r in t.itertuples()}
    fc = None
    if a.read_meta:
        rm = pd.read_csv(a.read_meta, sep='\t')
        fc = dict(zip(rm['read_id'], rm['flowcell']))
    hap = None
    if a.haplotype and Path(a.haplotype).exists():
        h = pd.read_csv(a.haplotype, sep='\t')
        hap = {(r.window, r.read_id): int(r.hap) for r in h.itertuples()}

    groups = sorted(set(win)) if a.per_window else ['ALL']
    rows, site_rows, read_rows = [], [], []
    for g in groups:
        keep = np.ones(len(si), bool) if g == 'ALL' else (win == g)
        if keep.sum() == 0:
            continue
        splits = {}
        for s in np.unique(si[keep]):
            m = keep & (si == s)
            if m.sum() < a.min_reads:
                continue
  # bag over seeds: each seed's +-1 split is sign-aligned to the first
  # (the sign is arbitrary) and averaged, so a site the GMM splits
  # inconsistently contributes a damped vector rather than a confident
  # but arbitrary one.
            acc = None
            n_seeds = 1 if a.split_method == 'pc1' else a.n_seeds
            for seed in range(n_seeds):
                sp = site_split(emb[m], seed=seed, method=a.split_method)
                if sp is None:
                    continue
                if acc is None:
                    acc = sp.copy()
                else:
                    acc += sp if np.dot(sp, acc) >= 0 else -sp
            if acc is not None:
                splits[int(s)] = (list(rid[m]), acc / n_seeds)
        if len(splits) < 3:
            continue
        reads = sorted({r for v in splits.values() for r in v[0]})
        score, part, load, lam = consensus(splits, reads)

  # ── scoring. Labels enter ONLY here. ────────────────────────────────
  # The positive class is taken from the data, not hardcoded: this ran on
  # COLO829 ('tumor'/'normal') long enough that the literal was invisible,
  # and on any other pairing (SPO1 'unmod_pcr'/'mod_native') it silently
  # mapped every read to 0 — frac_pos 0.0, ARI 0.0, AUROC NaN, which looks
  # like a clean null rather than a broken comparison.
        r2g = dict(zip(rid, grp))
        classes = sorted({str(r2g[r]) for r in reads})
        if len(classes) != 2:
            print(f'[{g}] skipped: {len(classes)} groups present {classes}')
            continue
        pos = a.pos_group if a.pos_group in classes else classes[1]
        ytrue = np.array([1 if str(r2g[r]) == pos else 0 for r in reads])
        ari = adjusted_rand_score(ytrue, part)
        auroc = roc_auc_score(ytrue, score) if len(set(ytrue)) > 1 else np.nan
        auroc = max(auroc, 1 - auroc)
        rec = dict(group=g, n_sites=len(splits), n_reads=len(reads),
                   eig=lam, ari_vs_sample=ari, auroc_vs_sample=auroc,
                   pos_class=pos, frac_pos=float(ytrue.mean()))
        if fc is not None:
  # within each sample, which flowcell did the read come from? if the
  # discovery is really a batch fingerprint this is where it shows.
            for samp in classes:
                sub = [i for i, r in enumerate(reads) if str(r2g[r]) == samp]
                fv = np.array([fc.get(reads[i], '?') for i in sub])
                if len(set(fv)) > 1 and len(sub) >= 8:
                    rec[f'ari_vs_flowcell_{samp}'] = adjusted_rand_score(
                        fv, part[sub])
        if hap is not None:
            hv = np.array([hap.get((g, r), -1) for r in reads])
            ok = hv >= 0
            rec['n_phased'] = int(ok.sum())
            rec['ari_vs_haplotype'] = (adjusted_rand_score(hv[ok], part[ok])
                                       if ok.sum() >= 8 and len(set(hv[ok])) > 1
                                       else np.nan)
        rows.append(rec)

        if a.emit_reads:
  # score sign is arbitrary (it is an eigenvector); ytrue is kept
  # alongside so a downstream orientation test can be honest about
  # which direction it chose and why.
            for i, r in enumerate(reads):
                read_rows.append(dict(group=g, read_id=r, score=float(score[i]),
                                      side=int(part[i]), sample=str(r2g[r]),
                                      is_pos=int(ytrue[i])))

        for s, l in load.items():
            cls = truth.get(('chr20', int(s_pos[s])), 'not_selected') if truth else ''
            site_rows.append(dict(group=g, window=str(s_win[s]),
                                  pos=int(s_pos[s]), loading=l,
                                  n=len(splits[s][0]), truth_cls=cls))

    res = pd.DataFrame(rows)
    sites = pd.DataFrame(site_rows)
    res.to_csv(out / 'discovery_summary.tsv', sep='\t', index=False)
    sites.to_csv(out / 'discovered_sites.tsv', sep='\t', index=False)
    if a.emit_reads:
        pd.DataFrame(read_rows).to_csv(out / 'discovered_reads.tsv',
                                       sep='\t', index=False)
        print(f'wrote {len(read_rows):,} read rows -> discovered_reads.tsv')
    print(res.round(3).to_string(index=False))
    if truth is not None and len(sites):
        print('\nsite loading by truth class (evaluation only):')
        print(sites.groupby(['group', 'truth_cls'])['loading']
              .agg(['median', 'size']).round(3).to_string())
    print(f'\nwrote {out}/discovery_summary.tsv and discovered_sites.tsv')


if __name__ == '__main__':
    main()
