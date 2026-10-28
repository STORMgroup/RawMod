# Checkpoints

Every file is a PyTorch state dict of the same 230,697-parameter model
(`rawmod/model.py`). The saved state also carries the normalization statistics and
the Deep SAD centre, which are not parameters and are not used when scoring. The
model is trained with the recipe in the top-level README
([Training](../README.md#training)). Load any of them with
`--checkpoint <path>` in the scoring scripts; all of them require `--normalize`
(MAD-normalized signal) and base identity visible in the pileup (`PILEUP_MASK_BASES=0`,
the default).

| Path | Trained on | Use |
|---|---|---|
| `rawmod.pt` | All data | Scoring new data. This is the model behind the genome map and the read-origin analyses of the paper. |
| `region_holdout/mixed.pt` | All data except one contiguous ~15% region of every genome, together with every read that overlaps it | The RawMod column of the benchmark (Table 1) and the first row of Table 2. Scored only inside the withheld regions. |
| `region_holdout/loco_{5mC,5hmC,6mA,4mC,5hmU}.pt` | As `mixed.pt`, and without every organism that carries the named modification | Zero-shot rows of Table 2: detection of a modification never seen in training. |
| `leaveout/<name>.pt` | As `mixed.pt`, and without every sample of one organism or library | Supplementary leave-out table of the benchmark (`test/evaluation/3_leaveout.sh`). |

`region_holdout/holdout_regions.tsv` lists the withheld region of every genome
(contig, 0-based start, exclusive end): one contiguous ~15% block per bacterial,
phage and plant genome, five of the 32 ONT oligonucleotide constructs, and human
chr1. The region-holdout models never trained on these regions or on any read that
overlaps them, and `scripts/evaluate/score_holdout_region.py` scores them only there
(`rawmod.paths.HOLDOUT_REGIONS`; `scripts/ground_truth/select_holdout_regions.py`
drew them). The benchmark's per-sample BEDs (`test/data/holdout/regions/`) are cut
from the same table.

`region_holdout/metrics.tsv` lists AUROC, AUPRC and the number of modified and
scored positions of each region-holdout model inside the withheld regions, for
the shipped training seed. The paper reports the mean over three seeds.

## Leave-out models

| Model | Withheld from training | Benchmark rows it scores |
|---|---|---|
| `ecoli.pt` | *E. coli* K-12 (wild type, dam- dcm-, dam- dcm- + M.SssI) | `ecoli_dam_6mA_reg`, `ecoli_dcm_5mC_reg`, `ecoli_mssi_5mC_reg` |
| `anabaena.pt` | *Anabaena* PCC 7120 | `anabaena_6mA_reg` |
| `hp26695.pt` | *H. pylori* 26695 (native and WGA) | `hp26695_4mC_reg`, `hp26695_6mA_reg` |
| `plant.pt` | *A. thaliana* | `arabidopsis_cpg_chr4`, `arabidopsis_noncpg_chr4` |
| `mammal.pt` | HG001 and HG002 | `hg002_cpg` |
| `ont5mC.pt` | ONT 5mC oligonucleotides and their control | `syn_5mC` |
| `ont5hmC.pt` | ONT 5hmC oligonucleotides and their control | `syn_5hmC` |
| `ont6mA.pt` | ONT 6mA oligonucleotides and their control | `syn_6mA` |
| `spo1.pt` | SPO1 (all barcodes) | `spo1_5hmU_reg` |
| `j99tden.pt` | *H. pylori* J99 and *T. denticola* | `hpj99_*_pb`, `tden_*_pb` (additional rows with PacBio ground truth) |

For 4mC (`hp26695.pt`), 5hmC (`ont5hmC.pt`) and 5hmU (`spo1.pt`) the withheld
library is the only training source of that modification, so these scores are
zero-shot detection of an unseen modification on a different test set than
Table 2. The model names used by the benchmark configuration
(`test/benchmark/config/rawmod_models.tsv`) are `rawmod` (= `region_holdout/mixed.pt`),
`released` (= `rawmod.pt`) and `logo_<name>` for the
leave-out models.
