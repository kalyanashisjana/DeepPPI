# Results and interpretation

## Provenance

The values below were checked against **CSV files and manifests inside the uploaded `PPI-Deep-Learning-Drug-Design.zip`**, supplemented where indicated by the user's previously shared terminal output. "Valid" means parseable/sanitizable molecular SMILES, **not validated inhibition of the gastric proton pump**.

## Dataset and training summary

| Dataset or run | Count / setting | Provenance |
|---|---:|---|
| Curated ChEMBL 37 train | 2,462,166 | `data/full_chembl37/manifest.json` |
| Curated ChEMBL 37 validation | 137,158 | Same manifest |
| Curated ChEMBL 37 test | 136,279 | Same manifest |
| PPI reference structures | 248 | `PPI_molecule_summary_248.csv` |
| PPI-related source candidates | 802 | S4 preprocessing manifest |
| Transformer fine-tuning structures | 786 | Recorded preparation run |
| S4 omeprazole-related fine-tune | 771 (694/77 train/val) | `s4_ppi786/data/manifest.json` |
| S4 reference-set fine-tune | 247 (222/25 train/val) | `s4_ppi248/data/manifest.json` |
| Custom Transformer | 6 layers, 8 heads, 256 features | `transformer/13_...py` |
| Pretrained published S4 | 3,310,663 parameter elements | Checkpoint inspection output |

## Generation: primary comparison

| Model / run | Generated | Valid | Unique valid | Validity | Unique among valid |
|---|---:|---:|---:|---:|---:|
| PPI-fine-tuned Transformer (reported main output) | 1,000 | 988 | 820 | 98.8% | 83.0% |
| S4-771 at T=1.00 | 1,000 | 959 | 952 | 95.9% | 99.3% |
| S4-247 (reported main output) | 1,000 | 957 | 957 | 95.7% | 100.0% |

Transformer counts were recalculated from `generation/transformer_ppi_finetuned_evaluated.csv`; the S4 counts match the previously supplied generation-terminal records. Training histories are present in the archive; complete S4 generation logs may not be contained in every exported subset.

## Sampling-temperature study: in-house Transformer

**Directly recalculated from uploaded `generation/transformer_temp_*.csv` files:**

| Sampling T | Generated | Valid | Unique valid | Validity | Unique / valid | Broad-motif matches (rows; not deduplicated) |
|---|---:|---:|---:|---:|---:|---:|
| 0.70 | 1,000 | 1,000 | 636 | 100.0% | 63.6% | 165 |
| 0.85 | 1,000 | 996 | 808 | 99.6% | 81.1% | 176 |
| 1.00 | 1,000 | 987 | 886 | 98.7% | 89.8% | 165 |

The broad-motif column is the row-level flag already present in the CSV, **not** the number of unique core structures. Do not equate these row counts to strict classical PPI-core retention.

## Sampling-temperature study: S4-771

**From previously shared generation-terminal output:**

| Sampling T | Generated | Valid | Unique valid | Validity | Unique / valid |
|---|---:|---:|---:|---:|---:|
| 0.70 | 1,000 | 989 | 928 | 98.9% | 93.8% |
| 0.85 | 1,000 | 984 | 963 | 98.4% | 97.9% |
| 1.00 | 1,000 | 959 | 952 | 95.9% | 99.3% |

The highest observed absolute count of unique valid S4 molecules was **963 at T=0.85**. This is not proof of optimum biological activity or optimum motif retention.

## S4 fine-tuning losses

The final entries in uploaded histories are:

| Experiment | Epochs | Final train loss | Final validation loss |
|---|---:|---:|---:|
| S4-771 | 15 | 0.209390 | 0.189858 |
| S4-247 | 15 | 0.243331 | 0.250420 |

Do not directly compare these losses against Transformer losses without aligning representations and objectives.

## Broad-core Transformer library

The uploaded `generation/unique_core_molecules/PPI_116_cheminformatics.csv` contains **116** selected unique generated structures with the broad sulfoxide–benzimidazole motif. Its saved fields indicate:

| Property | Result |
|---|---:|
| Selected broad-core unique molecules | 116 |
| Distinct saved Murcko-scaffold strings | 48 |
| Exact matches to characterized PPI structures | 8 |
| Exact matches to the curated ChEMBL training corpus | 18 |

An exact-match flag depends on the standardization rules used by its producing script. The **116 are broad motif-positive**, not automatically 116 classical proton pump inhibitors. Strict pyridylmethyl–sulfoxide–benzimidazole connectivity has not been conclusively audited across all model outputs. Activity prediction is not biochemical validation.

## Interpretation and caveats

- Both architectures can generate many valid molecules; their novelty, motif retention, and distribution differ.
- S4's ChEMBL v31 pretraining and the custom Transformer's ChEMBL 37 pretraining are not matched. Vocabulary, length limits, stereochemistry treatment, and sampling algorithms also differ.
- The reported numbers describe **individual generation batches**; inference variation should be measured with repeated independent random seeds for statistical claims.
- The GINE auxiliary model predicts **pIC50 estimates** from ligand graphs. Inhibitory activity cannot be inferred from syntactic SMILES validity alone. Fine-tuning and applicability-domain checks must be considered, especially for generated scaffolds dissimilar to the training/reference compounds.
- `04_PPI_Model_Comparison.ipynb` includes exploratory GINE prediction and chemical-analysis cells. Avoid adding a standalone prediction script unless it is separately implemented and tested.

## Recommended final presentation figures

1. Transformer vs S4: valid and unique valid counts (same 1,000-sample budget).
2. Effect of temperature on validity and uniqueness, with separate panels for each model.
3. Representative generated molecular structures, explicitly labeled as *generated candidates*.
4. Optional: S4 training/validation loss curves and Transformer checkpoint training history.

For actual commands and data origins, see [`commands.md`](commands.md) and [`dataset_description.md`](dataset_description.md).
