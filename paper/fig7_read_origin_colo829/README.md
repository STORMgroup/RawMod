# Label-free read-origin typing on COLO829

This directory reproduces the COLO829 part of Figure 5 and Table 2: RawMod's
per-read embeddings separate melanoma reads (COLO829) from the matched
lymphoblastoid reads (COLO829BL) in 10 kb windows of chr20, without labels and
without methylation calls, and the separation grows with the number of
differentially methylated CpGs in the window.

Run the commands below from this directory (`paper/fig7_read_origin_colo829/`).

## Data

The reads are from Oxford Nanopore's open COLO829 dataset
(`s3://ont-open-data/colo829_2024.03/`, also served over HTTPS at
`https://ont-open-data.s3.amazonaws.com/colo829_2024.03/`), four PromethION flow
cells: tumour PAU59949 and PAU61426, normal PAU59807 and PAU61427. Each flow cell
is about 1.2 TB of POD5 and the reads of one window are scattered over
thousands of files, so do not download the flow cells. `fetch_reads.py` reads
the published BAM to find which reads cover the windows, then pulls only those
reads' signal from the remote POD5 files with HTTP range requests:

```bash
python fetch_reads.py --sites inputs/sites_phase3.tsv \
  --bams tumor:PAU59949:<bam>,tumor:PAU61426:<bam>,normal:PAU59807:<bam>,normal:PAU61427:<bam> \
  --out-dir work/phase3 --per-group 18 --strand +
```

`--sites` lists the window centres (`inputs/sites_phase*.tsv`, one row per
window); `--bams` names the published aligned BAM of each flow cell
(`colo829_2024.03/flowcells/<flowcell>/...` in the bucket), as
`group:flowcell:path` triples. The script keeps, per window and group, up to
`--per-group` forward-strand reads that fully cover the window with a margin,
and writes their signal to `work/phase3/pod5/`. The reference is GRCh38
(`GCA_000001405.15_GRCh38_no_alt_analysis_set`).
The ranged reads go through `remote_pod5.py`, which ships next to the script: it
fetches only the header, footer and the signal batches that hold the wanted reads
of each remote POD5 file.

## Included inputs

| File | Content |
|---|---|
| `inputs/sites_phase2.tsv`, `inputs/sites_phase3.tsv` | The window centres, as `fetch_reads.py` reads them |
| `inputs/windows_phase2.tsv`, `inputs/windows_phase3.tsv` | The 54 chr20 windows of 10 kb (`name chrom start end`), in the two batches they were drawn in: 24 random windows with at least 20 covered CpGs, then 30 more (20 from the 5% of chr20 windows with the most CpGs, 10 random) |
| `inputs/windows_phase*_truth.tsv` | Per window: the number of differential CpGs (bulk methylation differing by at least 50 percentage points between the lines, at least 25 calls in each), used only to score the result |
| `inputs/reads_phase*.tsv` | Every read used, with its line (`tumor`/`normal`), flow cell and source POD5 file (`read_id group flowcell pod5`) |

## Steps

1. **Basecall and refine** the fetched reads with the genome-map pipeline
   (`scripts/genome_map/submit.sh --pod5 work/phase3/pod5 --ref GRCh38.fa --out work/phase3 --until preprocess`),
   which gives `reads_refined.bam` and `peaks_refined.tsv`.

2. **Per-read embeddings at every CpG of every window** (no labels used):

   ```bash
   python emb_all_reads.py --pod5 work/phase3/pod5 --bam work/phase3/preprocess/reads_refined.bam \
     --peaks work/phase3/preprocess/peaks_refined.tsv --windows inputs/windows_phase3.tsv \
     --read-groups inputs/reads_phase3.tsv --ref GRCh38.fa --ckpt ../../checkpoints/rawmod.pt \
     --level-table ../../rawmod/extern/uncalled_r1041_model_only_means.txt --site-type cpg --strand + \
     --min-reads 12 --normalize --output work/colo3_rawmod.npz
   ```

   Forward-strand reads only: pooling strands makes the consensus recover the
   strand instead of the sample.

3. **Label-free partition and scoring:**

   ```bash
   python discover.py --npz work/colo3_rawmod.npz --out-dir work/colo3_discovery \
     --per-window --split-method pc1 --min-reads 12 --emit-reads --read-meta inputs/reads_phase3.tsv
   ```

   At every CpG the reads are split by the sign of the first principal component
   of their embeddings; a consensus over the CpGs of the window gives one
   partition of its reads. Labels enter only in the last step, which reports the
   ARI of the partition against the true read origin and the oriented AUROC of
   the read score, per window (`discovery_summary.tsv`).

4. **Controls and direction** (optional): `fcnull_subset.py` relabels one line's
   reads by flow cell for the batch control; `ground_methyl.py` checks that the
   partition's direction agrees with methylation measured on the other flow-cell
   pair.

5. **Figures:** `fig_tumor_normal_joint.py` (panels a–d) and
   `fig_method_comparison.py` (panels e–h) read the discovery tables;
   `fig_readorigin.py` combines them into Figure 5. The Dorado, MethylBERT
   and ROCIT arms of panels e–h need those tools' outputs on the same reads.

   Steps 1–4 are not needed to redraw the figure: the discovery tables, the
   per-method AUROC and the embeddings of the example window are shipped in
   `paper/data/work/read_origin/`, so

   ```bash
   python fig_readorigin.py
   ```

   writes the figure to `paper/figures/` straight from a clone. The published
   figure also carries the HS578 pair and its flow-cell control, which are not
   shipped; those panels are left out and the rest are relabelled.

Run steps 1–3 for `phase2` as well; the paper pools both batches (54 windows).
