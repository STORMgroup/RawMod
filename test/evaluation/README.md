# Evaluation

We assume the datasets are downloaded ([`../data`](../data/README.md)) and the tools are configured
([`../README.md`](../README.md#configuration)). Run the scripts from the repository root, in this order:

| Script | What it does | Output |
|---|---|---|
| `train_unimeth_ft_region.sh` (optional) | fine-tunes UniMeth on RawMod's training positions (5hmC, 4mC, 5hmU) without the held-out regions; the UniMeth-FT columns of Table 1 | `$UNIMETH_FT_REGION_CHECKPOINTS/<mod>/final.pt` (default `$BENCH_WORK/unimeth_ft_region`) |
| `train_unimeth_agn.sh` (optional) | fine-tunes the modification-agnostic UniMeth-Agn; `UNIMETH_AGN_REGION=1 UNIMETH_AGN_MOD_TOKEN=1` gives the Table 1 column | `$UNIMETH_AGN_CHECKPOINTS`, `$UNIMETH_AGN_REGION_CHECKPOINTS` |
| `train_unimeth_ft.sh` (optional) | as `train_unimeth_ft_region.sh`, on the released model's training positions (supplementary positions table) | `$UNIMETH_FT_CHECKPOINTS/<mod>/final.pt` |
| `1_run_benchmark.sh [--slurm] [--tools all]` | read subsets, basecalls, read selection, rows, refinement, every tool, per-site aggregation | `$BENCH_WORK/<sample>/` |
| `2_score.sh` | nulls, tables and figures inside the held-out regions (Table 1), at held-out positions and at all sites | `$BENCH_WORK/results_<tag>/rawmod/` |
| `3_leaveout.sh [--slurm]` | the leave-out checkpoints on every sample, and the supplementary leave-out table | `tables/benchmark_leaveout.tex` in the held-out results folder |

`--tools all` adds every configured extra column (UniMeth-FT, UniMeth-Agn, Uncalled4, NovaMod) to the default tools
(RawMod, Dorado, UniMeth, DeepMod2, Rockfish, MethyNano). Every step skips work that is already done, so a script can be rerun after a failure. With `--slurm`, each
task is one SLURM job, chained by dependencies; run the next script when the jobs have finished.
