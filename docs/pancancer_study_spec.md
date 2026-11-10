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

## Paper-style design (`src/pancancer_dataset.py`), the one requested for the experiments
raw snapshot -> positive / negative (`exp_negative`) -> CD-HIT 0.4 once over all groups -> length <= 35 -> 80:20.
- colorectal, lung, melanoma: each its own stratified 80:20 (5-fold CV on train in the runner).
- pooled: train = the three trains, test = the three tests (24:6 is 80:20 by construction).
- other cancers (and hematologic): test only, `test_set = other_<disease>`; report a disease only with >= 50 positives and negatives.
- Cost: CD-HIT keeps one peptide per cluster, so positives drop a lot (paper: 671 -> 303); `table1_stages.csv` shows it.
The cluster-fold arms (`pancancer_split.py`) remain as the alternative that keeps all peptides.
