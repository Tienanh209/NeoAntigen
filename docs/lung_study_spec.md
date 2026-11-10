# Lung cancer neoantigen prediction: study specification (guide steps 1, 3-5)

**Task.** Peptide sequence -> is it a T cell-validated lung-cancer neoantigen epitope?
**Prediction unit.** One peptide (8-35 aa). **Primary metric.** AUPRC (also MCC, AUROC, sensitivity/specificity, F1).
**Intended use.** Prioritising candidate peptides; sequence-only, no HLA/patient context, not a clinical claim.

## Labels (CEDAR, linear peptides; `src/lung.py`)
| tier | rule | used as |
|---|---|---|
| `positive` | neoantigen from a lung cancer disease term, T cell assay with a Positive* result | class 1 |
| `exp_negative` | same pool, T cell assay, no positive result | class 0 only when `exp_negative=True` (strict-negative comparison) |
| `bg_viral`, `bg_taa` | non-neoantigen viral / tumour-associated epitopes | class 0, **background negatives** (not experimentally tested on lung neoantigen context) |
| `unknown` | lung neoantigen without any T cell assay | excluded, counted, never turned into a negative |

Conflicts: a sequence seen in both classes is dropped (`ambiguous_dropped`). Non-standard residues are dropped.

## Splits (`src/lung_split.py`, T = 2022, E = 2026)
- Time decides first: first public evidence year in (T, E] -> temporal; <= T (and undated) -> development.
- Temporal peptides already in development are *new annotations of known sequences* and are excluded (listed in `split_exclusions.csv`).
- Development: CD-HIT 0.4, then 80/20 split by cluster -> `train` / `test` (conventional independent, locked).
- `temporal_lowsim`: temporal peptides with max identity to development < 0.4. It overlaps `temporal_all`.
- Cross-split similarity is re-measured without CD-HIT (`similarity_report.csv`).

## Known limits
- Dates are the earliest CEDAR reference **year** (no month); undated rows never enter the temporal set.
- Temporal negatives are background viral/TAA epitopes dated in (T, E], so prevalence differs from development.
- Similarity screen = shared 5-mer + ungapped identity; run `cd-hit-2d` on Colab for the alignment-based answer.
- Few temporal positives: report the limitation rather than re-labelling another holdout as temporal.
