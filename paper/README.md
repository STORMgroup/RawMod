# Reproducing the results of the paper

Each directory holds the scripts that produced one result, as they were run. The
benchmark against other tools (Table 1, Figure 3) lives in [`test/`](../test/README.md),
which is a complete, self-contained pipeline. The scripts here are the analysis
and figure code of the remaining results.

The numbers behind every figure and table are shipped in [`data/`](data/), so the
figure and table scripts run on a fresh clone with no configuration:

```bash
python paper/table2_fig4_generalization/make_region_figures.py table   # Supplementary Table S1
python paper/table2_fig4_generalization/make_fig4_combined.py           # Figure 4
python paper/fig6_dinoflagellate_map/make_dino_figures.py              # Supplementary Figures S5 and S6
```

Figures are written to `paper/figures/` and LaTeX tables to `paper/tables/`.
The steps that read raw signal or the training features (the `eval_*`, `score_*`
and analysis scripts) need the data itself; they are listed per result below.

Common helpers: `common/make_paper_figures.py` (palette, figure sizes, `save()`).
Every figure script imports it and puts `paper/common` on the path itself, so the
scripts run from anywhere.

| Paper element | Directory | Script(s) | Inputs |
|---|---|---|---|
| Figure 1b, signal contrast of each chemistry | `fig1_signal_contrast/` | `signal_contrast.py` writes the per-position contrasts (`signal_contrast.npz`, shipped); `common/make_paper_figures.py overview` draws the figure | Pileup features (`features.h5`) of a modified library and its unmodified control at matched positions |
| Supplementary Table S1 and Figure 4a (zero-shot detection of withheld modifications) | `table2_fig4_generalization/` | `../scripts/evaluate/score_holdout_region.py` scores a model inside the withheld regions; `make_region_figures.py table` writes the table; `make_fig4_combined.py` draws Figure 4 (all four panels); `make_region_sites_table.py` the test-set composition (Supplementary Table S2) | Training feature tree, `checkpoints/region_holdout/*.pt`, the withheld regions, `checkpoints/region_holdout/holdout_regions.tsv` (drawn by `scripts/ground_truth/select_holdout_regions.py`) |
| Zero-shot 4mC by sequence context (motif counts, AUROC per motif and away from 6mA) | `table2_fig4_generalization/` | `check_4mc_context.py SCORES_DIR hpylori_26695.fa` | The per-site score dumps of `region_sites_analyze.py` and the *H. pylori* 26695 reference |
| Figure 4b (accuracy against read depth) | `table2_fig4_generalization/` | `eval_read_depth_region.py` | As above |
| Figure 4c and 4d and Supplementary Figure S1 (labelled sites of a withheld modification) | `table2_fig4_generalization/` | Training: `scripts/train/run_matched_loco.py` with the few-shot options (`_fewshot` in that file); figures: `make_fig4_combined.py` (Figure 4c, 4d) and `make_region_figures.py fewshot` (Supplementary Figure S1), which read the per-run metrics shipped in `data/runs/fewshot_finetune/` and `data/runs/fewshot_pool/` | Training feature tree (training only; the figure runs on a fresh clone) |
| Supplementary Tables S3 to S5: ablations, auxiliary-loss comparison, no-learning baseline | `supp_ablations/` | `make_ablation_table_region.py`; `auxloss_embed_arms.py` → `auxloss_metrics.py` → `auxloss_loco_boot.py` → `make_auxloss_table_region.py`; `baseline.py` → `make_table.py` | Ablation checkpoints (one per configuration; available on request) and the training feature tree |
| Supplementary Figure S2 (embedding of benchmark positions) and the embedding statistics | `figS2_embedding/` | `eval_embedding_bench_region.py` → `make_embedding_bench_fig_region.py`; `eval_embedding_region.py` for the test-set statistics | Benchmark pileups (`test/`), `checkpoints/region_holdout/mixed.pt` |
| Supplementary Figures S5 and S6 and Tables S6 and S7, 5hmU map of the *A. carterae* contig | `fig6_dinoflagellate_map/` | Scoring: `scripts/genome_map/submit.sh --base T` with `checkpoints/rawmod.pt`; analysis: `run_analysis.sh` runs `dino_analysis.py`, `dino_gaps.py`, `dino_repeats.py`, `dino_alignment_confound.py`, `dino_literature.py` and the `strand/` scripts (`array_blocks.py`, `gc_matched_null.py`, `genestrand_and_context.py`, `strand_concordance.py`); figures and tables: `make_dino_figures.py`, `make_dino_tables.py` | The contig FASTA, its gene annotation (GFF3), RepeatMasker output, and the reads of the contig (to be published with the data) |
| Figure 5 and Table 2, label-free read-origin typing | `fig7_read_origin_colo829/` | See its [README](fig7_read_origin_colo829/README.md): a complete walkthrough on the public COLO829 data | COLO829 raw signal from ONT open data; the window and read lists are included in `inputs/` |
| Supplementary Table S28, RawMod on sup, hac and fast basecalls with and without boundary refinement | `supp_basecallers/` | `make_table.py` from the output of `test/benchmark/bench/compare_inputs.py` (`BENCH_BASECALL_MODEL`, `BENCH_REFINE=none`; see the benchmark README) | The benchmark work tree with the extra basecalls; the scored comparison is shipped in `data/work/basecallers/` |
| Supplementary Table S10, SPO1 typing with models that did not train on these reads | `fig7_read_origin_colo829/` | `spo1_heldout_models.sh` (released, region-holdout, no-5hmU and no-SPO1 models through the SPO1 arm), then `spo1_heldout_table.py collect`; `spo1_heldout_table.py` prints the table | SPO1 reads per barcode (see the script header); the per-window AUROCs are shipped in `data/work/read_origin/spo1_heldout.tsv` |
| Supplementary Table S27, the cost of every caller on the same reads | `supp_cost/` | `tool_cost.sh` (Dorado basecall and four modification models, DeepMod2, Rockfish, UniMeth, RawMod on GPU and CPU; shared measurement code in `timing_lib.sh`); `make_cost_table.py` writes the table | A pod5 file and its reference, and the benchmark's tool settings (`test/benchmark/config/site.env`); the measured run is shipped in `data/work/cost/tool_cost.tsv` |
| Supplementary Section S10.4, memory against the number of parallel processes | `supp_cost/` | `memory_scaling.sh` (basecall, refine, then fused featurization + scoring with 1, 2, 4 and 8 processes on one GPU); `make_memory_table.py` | A pod5 file and its reference; the measured run is shipped in `data/work/cost/memory_scaling.tsv` |
| Supplementary Section S4, library invariance with and without the curriculum | `supp_ablations/` | `curriculum_library_invariance.py` (scores the benchmark's library-control images with each checkpoint; `--summary` prints the shipped result) | The benchmark work tree after `bench.library_control cands` and `rawmod`; the final and no-curriculum checkpoints (available on request); result shipped in `data/work/curriculum_libinv/` |
| Supplementary Table S23, operating points at one shared threshold (detection rate at 1% FPR per row, averaged over rows and over the five modification types) | `supp_threshold/` | `fixed_threshold.py compute` scores the benchmark work tree; `fixed_threshold.py` writes the table from the shipped result | The benchmark work tree after `2_score.sh` (`BENCH_HOLDOUT=region`); the result is shipped in `data/work/threshold/` |
| Supplementary Table S24, different modifications on the same sequence | `supp_same_sequence/` | `same_sequence.py` | The benchmark's per-site score files (`test/`, after `2_score.sh`) |
| Supplementary Table S25, Table 1 with confidence intervals | [`test/`](../test/README.md) | `python -m bench.ci` (cluster bootstrap over reference positions), then `python -m bench.ci_table` writes the table, from the shipped rows when the benchmark has not been run; `python -m bench.paired_diff` gives the paired RawMod against UniMeth-Agn interval quoted in the text | The benchmark work tree after `2_score.sh`; the bootstrap rows are shipped in `data/work/benchmark_ci/` |
| Supplementary Figure S3, modifications that no released caller detects | `supp_dark_mods/` | `run_dark.sh` runs Dorado's all-context calls on the *A. carterae* contig, selects sites by what Dorado says about them (`select_sites.py`), builds their pileups and embeds them with the released model (`embed_sites.py`); `make_dark_mods_fig.py` draws the figure | The contig reads (POD5) with their refined alignment, Dorado and modkit; the embeddings are shipped in `data/work/dark_mods/` and the benchmark embedding in `data/work/figures/` |
| Supplementary Figure S4, calibration of the site scores | `supp_calibration/` | `score_region_sites.py` scores the withheld-region test set with `checkpoints/region_holdout/mixed.pt`; `make_calibration_fig.py` draws the reliability diagram and writes the calibration statistics | Training feature tree (scoring only); the per-position scores are shipped in `data/work/calibration/` |
| Supplementary Figure S7, Dorado basecalling errors on 5hmU DNA | `supp_basecall_errors/` | `bc_strain_typing.py` (per-read error rates of native against PCR reads, length-matched AUROC), `profile_ties.py` (error rate against the distance to the nearest thymine), `window_auroc.py` (per-window AUROC of the Dorado statistics); `fig_basecall_errors.py` draws the figure | The SPO1 read-origin directories (one per barcode: `reads.tsv`, `windows.tsv` and the refined alignment) and the SPO1 reference; the per-read tables and profiles are shipped in `data/work/basecall_errors/` |

## Shipped results (`data/`)

| Path | Content |
|---|---|
| `data/runs/` | Per-run metrics of the region-holdout models: the mixed and zero-shot runs for three seeds, every ablation, the mixed-reference variant, and the few-shot runs (`fewshot_finetune/`, `fewshot_pool/`, one directory per fold, number of labelled sites and draw). One `metrics/<fold>.tsv` per run, as `score_holdout_region.py` writes it. |
| `data/work/figures/` | Inputs of the figure scripts: the signal contrast of Figure 1b, the read-depth sweep, the benchmark site embeddings and the few-shot curve. |
| `data/work/dinoflagellate/` | Outputs of the *A. carterae* analyses: per-class statistics, repeat and alignment caches, intergenic gaps and the strand-resolved tables. The score tracks themselves go with the sequencing data. |
| `data/work/read_origin/` | COLO829 read-origin runs (the discovered partition per window), the per-method AUROC of the COLO829 and SPO1 arms (the released model on the cell lines, the region-holdout model on SPO1), the per-window SPO1 AUROC of the models that did not train on these reads (`spo1_heldout.tsv`), and the per-read embeddings of the example window. |
| `data/work/auxloss/`, `data/work/same_sequence/`, `data/work/region_sites/`, `data/work/basecallers/`, `data/work/cost/`, `data/work/curriculum_libinv/`, `data/work/threshold/` | The supplementary comparisons: auxiliary losses, mixed chemistry on one sequence, test-set composition, basecallers with and without boundary refinement, the cost of every caller and memory against parallel processes, library invariance with and without the curriculum, and the operating points at one shared threshold. |
| `data/work/calibration/`, `data/work/basecall_errors/`, `data/work/dark_mods/`, `data/work/benchmark_ci/` | The per-position scores behind the calibration figure, the per-read error tables and profiles behind the basecalling-error figure, the *A. carterae* embeddings behind the dark-modification figure, and the bootstrap rows behind the confidence-interval table. |
| `data/COLO829/` | The windows of Figure 5 and their differential-CpG counts. |

The HS578 arm of Figure 5 is not shipped (see data availability below), and
neither is the flow-cell control, which is measured on that pair. The figure
scripts drop those panels and relabel the rest, so `fig_readorigin.py` run on the
shipped data gives the COLO829 and SPO1 panels of the published figure.

## Locating the data

Three environment variables override the defaults: `RAWMOD_RESULTS` (training
and evaluation runs, one directory per run; defaults to `paper/data/runs`),
`RAWMOD_WORK` (the outputs of these scripts; defaults to `paper/data/work`) and
`RAWMOD_DATA` (datasets: reads, references, annotations and training features,
which are not shipped). `RAWMOD_FIGS` and `RAWMOD_TABLES` change where figures
and tables are written.

## Data availability

The COLO829 reads are public (see `fig7_read_origin_colo829/`).
The HS578 cell-line pair of Figure 5 is not publicly available from us. The SPO1
reads (native and PCR libraries) are deposited under BioProject
[PRJNA1549975](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1549975), and the raw
signal (POD5) of the native and PCR libraries is also in [SquiDBase](https://squidbase.org) under
accession SQB000031.
The *A. carterae* reference is contig JBNFNT010000839.1 in GenBank. The *A. carterae*
reads are deposited under BioProject
[PRJNA1110228](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1110228), and the raw
signal (POD5) of the two flow cells we use is in SquiDBase under accession SQB000028.

The HS578 and SPO1 arms of Figure 5 use the same scripts as
COLO829, with `coverage_matched.py` added for SPO1 because its two arms differ in
read length.
