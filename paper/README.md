# Reproducing the results of the paper

Each directory holds the scripts that produced one result, as they were run. The
benchmark against other tools (Table 1, Figure 3) lives in [`test/`](../test/README.md),
which is a complete, self-contained pipeline. The scripts here are the analysis
and figure code of the remaining results.

The numbers behind every figure and table are shipped in [`data/`](data/), so the
figure and table scripts run on a fresh clone with no configuration:

```bash
python paper/table2_fig4_generalization/make_region_figures.py table   # Table 2
python paper/fig6_dinoflagellate_map/make_dino_figures.py              # Figure 6
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
| Table 2, Figure 4a (zero-shot detection of withheld modifications) | `table2_fig4_generalization/` | `../scripts/evaluate/score_holdout_region.py` scores a model inside the withheld regions; `make_region_figures.py table generalization` writes the table and figure; `make_region_sites_table.py` the supplementary site counts | Training feature tree, `checkpoints/region_holdout/*.pt`, the withheld regions, `checkpoints/region_holdout/holdout_regions.tsv` (drawn by `scripts/ground_truth/select_holdout_regions.py`) |
| Zero-shot 4mC by sequence context (motif counts, AUROC per motif and away from 6mA) | `table2_fig4_generalization/` | `check_4mc_context.py SCORES_DIR hpylori_26695.fa` | The per-site score dumps of `region_sites_analyze.py` and the *H. pylori* 26695 reference |
| Figure 4b (accuracy against read depth) | `table2_fig4_generalization/` | `eval_read_depth_region.py` | As above |
| Figure 5 and Fig. S-fewshot (labelled 5hmU sites) | `fig5_fewshot/` | Training: `scripts/train/run_matched_loco.py` with the few-shot options (`_fewshot` in that file); figure: `../table2_fig4_generalization/make_region_figures.py fewshot` | Training feature tree |
| Supplementary ablations, auxiliary-loss comparison, no-learning baseline | `supp_ablations/` | `make_ablation_table_region.py`; `auxloss_embed_arms.py` → `auxloss_metrics.py` → `auxloss_loco_boot.py` → `make_auxloss_table_region.py`; `baseline.py` → `make_table.py` | Ablation checkpoints (one per configuration; available on request) and the training feature tree |
| Figure S2 (embedding of benchmark positions) and the embedding statistics | `figS2_embedding/` | `eval_embedding_bench_region.py` → `make_embedding_bench_fig_region.py`; `eval_embedding_region.py` for the test-set statistics | Benchmark pileups (`test/`), `checkpoints/region_holdout/mixed.pt` |
| Figure 6, 5hmU map of the *A. carterae* contig | `fig6_dinoflagellate_map/` | Scoring: `scripts/genome_map/submit.sh --base T` with `checkpoints/rawmod.pt`; analysis: `run_analysis.sh` runs `dino_analysis.py`, `dino_gaps.py`, `dino_repeats.py`, `dino_alignment_confound.py`, `dino_literature.py` and the `strand/` scripts (`array_blocks.py`, `gc_matched_null.py`, `genestrand_and_context.py`, `strand_concordance.py`); figures and tables: `make_dino_figures.py`, `make_dino_tables.py` | The contig FASTA, its gene annotation (GFF3), RepeatMasker output, and the reads of the contig (to be published with the data) |
| Figure 7 and Table 3, label-free read-origin typing | `fig7_read_origin_colo829/` | See its [README](fig7_read_origin_colo829/README.md): a complete walkthrough on the public COLO829 data | COLO829 raw signal from ONT open data; the window and read lists are included in `inputs/` |
| Supplementary table, RawMod on sup, hac and fast basecalls with and without boundary refinement | `supp_basecallers/` | `make_table.py` from the output of `test/benchmark/bench/compare_inputs.py` (`BENCH_BASECALL_MODEL`, `BENCH_REFINE=none`; see the benchmark README) | The benchmark work tree with the extra basecalls; the scored comparison is shipped in `data/work/basecallers/` |
| Supplementary Table S14, SPO1 typing with models that did not train on these reads | `fig7_read_origin_colo829/` | `spo1_heldout_models.sh` (released, region-holdout, no-5hmU and no-SPO1 models through the SPO1 arm), then `spo1_heldout_table.py collect`; `spo1_heldout_table.py` prints the table | SPO1 reads per barcode (see the script header); the per-window AUROCs are shipped in `data/work/read_origin/spo1_heldout.tsv` |
| Supplementary Table S2, the cost of every caller on the same reads | `supp_cost/` | `tool_cost.sh` (Dorado basecall and four modification models, DeepMod2, Rockfish, UniMeth, RawMod on GPU and CPU; shared measurement code in `timing_lib.sh`); `make_cost_table.py` writes the table | A pod5 file and its reference, and the benchmark's tool settings (`test/benchmark/config/site.env`); the measured run is shipped in `data/work/cost/tool_cost.tsv` |
| Supplementary Section S2.2, memory against the number of parallel processes | `supp_cost/` | `memory_scaling.sh` (basecall, refine, then fused featurization + scoring with 1, 2, 4 and 8 processes on one GPU); `make_memory_table.py` | A pod5 file and its reference; the measured run is shipped in `data/work/cost/memory_scaling.tsv` |
| Supplementary Section S3, library invariance with and without the curriculum | `supp_ablations/` | `curriculum_library_invariance.py` (scores the benchmark's library-control images with each checkpoint; `--summary` prints the shipped result) | The benchmark work tree after `bench.library_control cands` and `rawmod`; the final and no-curriculum checkpoints (available on request); result shipped in `data/work/curriculum_libinv/` |
| Supplementary table, different modifications on the same sequence | `supp_same_sequence/` | `same_sequence.py` | The benchmark's per-site score files (`test/`, after `2_score.sh`) |

## Shipped results (`data/`)

| Path | Content |
|---|---|
| `data/runs/` | Per-run metrics of the region-holdout models: the mixed and zero-shot runs for three seeds, every ablation, and the mixed-reference variant. One `metrics/<fold>.tsv` per run, as `score_holdout_region.py` writes it. |
| `data/work/figures/` | Inputs of the figure scripts: the signal contrast of Figure 1b, the read-depth sweep, the benchmark site embeddings and the few-shot curve. |
| `data/work/dinoflagellate/` | Outputs of the *A. carterae* analyses: per-class statistics, repeat and alignment caches, intergenic gaps and the strand-resolved tables. The score tracks themselves go with the sequencing data. |
| `data/work/read_origin/` | COLO829 read-origin runs (the discovered partition per window), the per-method AUROC of the COLO829 and SPO1 arms, the per-window SPO1 AUROC of the models that did not train on these reads (`spo1_heldout.tsv`), and the per-read embeddings of the example window. |
| `data/work/auxloss/`, `data/work/same_sequence/`, `data/work/region_sites/`, `data/work/basecallers/`, `data/work/cost/`, `data/work/curriculum_libinv/` | The supplementary comparisons: auxiliary losses, mixed chemistry on one sequence, test-set composition, basecallers with and without boundary refinement, the cost of every caller and memory against parallel processes, and library invariance with and without the curriculum. |
| `data/COLO829/` | The windows of Figure 7 and their differential-CpG counts. |

The HS578 arm of Figure 7 is not shipped (see data availability below), and
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
The HS578 cell-line pair of Figure 7 is not publicly available from us. The SPO1
reads (native and PCR libraries) are deposited under BioProject
[PRJNA1549981](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1549981), and the native
SPO1 reads are also in [SquiDBase](https://squidbase.org) under accession SQB000027.
The *A. carterae* reference is contig JBNFNT010000839.1 in GenBank; the availability
of the *A. carterae* reads is given in the paper. The HS578 and SPO1 arms of Figure 7 use the same scripts as
COLO829, with `coverage_matched.py` added for SPO1 because its two arms differ in
read length.
