# Pan-cancer study spec (draft, numbers from the 2026-11-10 CEDAR snapshot)

Question: do T-cell-validated neoantigen signals transfer between cancers?

## Available data (unique sequences, before CD-HIT)
| group | positive | exp. negative | papers (pos / neg) |
|---|---|---|---|
| melanoma | 1534 | 5542 | 126 / 22 |
| lung | 644 | 1792 | 53 / 21 |
| colorectal | 462 | 8466 | 49 / 18 |
| other solid (pancreatic, breast, bladder, renal, hepatocellular, brain, ovarian_gyn, head_neck, stomach_esophageal, other) | ~1450 | ~3500 | - |
| bile_duct | 42 | 837 | too small to report alone |
| hematologic | 126 | 76 | kept apart, optional extra OOD set |

`multi` (85 positives) = peptides annotated with several cancers; excluded from every arm.

## Design
Arms: **CRC-only, melanoma-only, lung-only, pooled(3)**. Each is tested on (a) held-out part of each of the 3 diseases
and (b) every other solid cancer with >= 100 positives, reported per disease, plus a combined "other solid" AUC.

1. **Split by CD-HIT cluster (0.4), 5 outer folds, one fold = the test.** Grouping by paper was tried and is
   infeasible: rows cite several papers, so 83% of the core rows fall in ONE connected paper component. The paper
   effect stays a risk (CRC negatives: top paper holds 62% of them), so `summary.csv` reports `paper_share` per disease
   and a leave-top-paper-out sensitivity run is an open item. Report mean +- sd over the 5 outer folds; CRC has too few
   positives (~465) for one 20% test.
2. **Negatives: disease-matched experimental negatives are primary.** Viral/TAA negatives (the paper's choice) are kept as
   a secondary setting, because "virus vs human" is a shortcut that inflates cross-disease AUC. Report both.
3. **Class prior differs a lot per disease** (CRC 462 / 8466, bladder 172 / 11). Compare with ROC-AUC, and report
   prevalence next to every AUPRC. For training, subsample negatives to a fixed ratio (e.g. 1:5) per arm.
4. **Size control.** Melanoma is 58% of the pooled positives. Add a pooled arm downsampled to 3 x 462 so that
   pooled-vs-single is not just more data.
5. **OOD leakage filter.** Remove from every test set any sequence similar (CD-HIT, same threshold as the splits) to
   any training sequence of that arm. Report AUC before and after the filter.
6. **Multiple seeds** (>= 3) because the test sets are small.

## Open points
- Merge or drop the `other` group (186 positives; mixed bag).
- Whether to add hematologic as an extra OOD set.
- Whether to use `year_first` temporal test (see `lung_split.py`) in addition to grouped folds.

## Paper-style design (`src/pancancer_dataset.py`): comparison with the paper only
raw snapshot -> positive / negative (`exp_negative`) -> CD-HIT 0.4 once over all groups -> length <= 35 -> 80:20.
- colorectal, lung, melanoma: each its own stratified 80:20 (5-fold CV on train in the runner).
- pooled: train = the three trains, test = the three tests (24:6 is 80:20 by construction).
- other cancers (and hematologic): test only, `test_set = other_<disease>`; report a disease only with >= 50 positives and negatives.
- Cost: CD-HIT keeps one peptide per cluster, so positives drop a lot (paper: 671 -> 303); `table1_stages.csv` shows it.
The cluster-fold arms (`pancancer_split.py`) remain as the alternative that keeps all peptides.

## Main design (decided after the Colab run): cluster split, no peptide discarded
Paper-style CD-HIT representatives left 63 / 34 / 136 positives for colorectal / lung / melanoma (tests of 12 / 7 / 27
positives), too small to evaluate each cancer. The main layout therefore groups by CD-HIT cluster instead of removing
peptides (`pancancer_split.write_main_arms`):
- colorectal, lung, melanoma: each trained and tested on its own ~80:20 (one outer fold is the 20%); 5-fold CV on train
  with `split_mode="grouped"`, which now folds by the frame's `cluster_id`.
- pooled: the three trains together, tested on the three tests (24:6 = 80:20), plus the other cancers as test rows.
- Every test row carries `test_set` and `similar_to_train`; report on `~similar_to_train` and show the all-rows number
  beside it. All negatives are kept (`neg_ratio=None`); AUC is compared across diseases, prevalence is reported.

## Similarity at 30%
CD-HIT has an identity floor of 0.4 (the code asserts it). `pancancer_split.mmseqs_clusters` / `cluster_ids(method="mmseqs")`
cluster with MMseqs2 (`easy-cluster --min-seq-id 0.3 -c 0.8 --cov-mode 1 -s 8`), and `mmseqs_filter` gives CD-HIT-style
representatives for the paper-style pipeline. Use a new dataset name (e.g. `pancancer_cedar_v1_id30`); `meta.json` records
`cluster_method`, `sim_id` and `cluster_kw`. MMseqs2 is less sensitive on 8-35 aa peptides, so confirm with
`check_similarity` (independent, ungapped, a lower bound; not a proof of < 30%).

## Reproducibility rules
- The raw snapshot is the data: CEDAR changes over time, so a result is reproduced from `pancancer_cedar_raw_YYYYMMDD[_n].csv`,
  never from a fresh download. Snapshots are never overwritten (a second download the same day gets `_2`).
- Pin the snapshot in notebooks: `pancancer.get_raw(config.DATA_DIR, snapshot="20261110")`. Without it the latest file is
  taken and a warning is printed when several exist.
- The snapshot's `.meta.json` holds its SHA-256, the labelling rules (`disease_groups`, `not_cancer_re`), the `git_sha`
  and tool versions; `get_raw` warns if the file no longer matches the checksum.
- A dataset's `meta.json` (manifest) records `snapshot_file`, `snapshot_sha256`, `manifest_sha256`, `tool_versions`
  (python, numpy, pandas, scikit-learn, cd-hit, mmseqs), `git_sha`, the parameters and `seed`. The manifest itself stores
  `cluster_id` and `outer_fold`, so the split is frozen and need not be recomputed.
- `git_sha` is the checked-out commit: run builds from a clean checkout (as Colab does), not with local edits.

## Per-arm experiments (`embeddings_pancancer.ipynb`, `experiments_arm.ipynb`)
Any arm (`colorectal`, `lung`, `melanoma`, `pooled`) at any outer fold, every negative kept. Documented fold-0 sizes (train pos/neg | own test pos/neg):
colorectal 384/6,867 | 77/1,584; lung 547/1,480 | 97/311; melanoma 1,265/4,456 | 268/1,075; pooled 2,196/12,803 | 442/2,970.
- **Embed once.** Every arm frame is a subset of the manifest, so `embeddings_pancancer.ipynb` embeds the whole manifest once
  (cache key `ps.emb_id(manifest)`); `load_arm` attaches it as `df.attrs["emb_id"]` and the runner reads that cache. Run identity still uses
  the frame's own `build_id`, so arms never collide in the results store.
- **Switch arm** by editing `ARM, FOLD` in block 1 of `experiments_arm.ipynb`; experiment ids are prefixed per arm (`CRC_`, `LUNG_`, `MEL_`, `POOL_`).
  `ps.build_arm` rebuilds the frame from `manifest.csv` (deterministic) instead of trusting an old directory.
- The runner scores every non-train row: read tables with `subset=OWN` (own test) and `thr_on=OWN` (operating point fixed on it);
  other diseases come from `ps.eval_uids(df)` through `results.by_set` (rows similar to the arm's train are removed). For `pooled`, `OWN` is the
  three tests together and each disease also has its own "(own test)" column.
- `results.cluster_bootstrap` / `bootstrap_table` resample whole clusters, because peptides of one cluster are near-duplicates.
- `write_arm` adds `_nr<r>` to the directory when negatives were capped, so a 5x frame never overwrites the full one.
