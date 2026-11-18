# Dataset description

## Scope and provenance

This project uses chemical structures from **ChEMBL 37** for the in-house GINE and causal Transformer workflows, alongside the **published S4 checkpoint pretrained on ChEMBL v31**. The S4 pretrained corpus is not identical to the ChEMBL 37 corpus. For the target-specific experiments, a ChEMBL-derived omeprazole-related series and a smaller experimentally characterized proton pump inhibitor (PPI) reference collection are used.

### Sources and local files

| Resource | Local relative path | Purpose |
|---|---|---|
| Raw ChEMBL 37 structures | `data/raw/chembl_37_chemreps.txt.gz` | Starting point for corpus curation |
| Full-corpus manifest | `data/full_chembl37/manifest.json` | Exact filtering and split counts |
| Initial sampled-pretraining manifest | `data/processed/chembl37_pretrain_manifest.json` | Earlier 100k-molecule experiment |
| Experimental PPI summary | `PPI_molecule_summary_248.csv` | 248 structure entries with associated assay data |
| PPI activity measurements | `PPI_records_with_pIC50_309.csv` | 309 curated activity-record entries |
| Corrected analogue search | `analysis/omeprazole_series/full_chembl37_corrected_matches.csv` | 802 source entries for target fine-tuning |
| Transformer PPI output | `generation/transformer_ppi_finetuned_evaluated.csv` | Generated SMILES and evaluation fields |
| S4-771 preparation | `s4_fine_tuning/experiments/s4_ppi786/data/manifest.json` | Fine-tuning compatibility/rejections |
| S4-247 preparation | `s4_fine_tuning/experiments/s4_ppi248/data/manifest.json` | Fine-tuning compatibility/rejections |

The downloaded source is the ChEMBL 37 chemical representations file:
`https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/chembl_37_chemreps.txt.gz`.

## Full ChEMBL 37 preparation

Script: `data_preparation/10_prepare_full_chembl.py` (also present as a root-level copy in the source archive).

The procedure parses SMILES with RDKit, selects the largest organic fragment, retains molecular structures with **5–80 heavy atoms**, restricts elements to **H, B, C, N, O, F, P, S, Cl, Br, I**, preserves charges and specified stereochemistry, restricts SMILES to **200 characters**, excludes exact standardized PPI structures, and deduplicates canonical isomeric SMILES. A deterministic SHA-256 bucket allocates approximately 90%/5%/5% to training/validation/test.

**Verified counts** (from `data/full_chembl37/manifest.json`):

| Stage | Records |
|---|---:|
| ChEMBL input rows | 2,897,819 |
| Invalid/filtered rows | 54,740 |
| Excluded exact PPI structures | 377 |
| Duplicate standardized SMILES | 107,099 |
| Train | 2,462,166 |
| Validation | 137,158 |
| Test | 136,279 |
| **Total retained** | **2,735,603** |

**Important:** The earlier `01_ChEMBL_Data_Preparation.ipynb` and `data/processed/chembl37_pretrain_manifest.json` describe an initial, *different* experiment: a 100,000-molecule dataset with 90,000 training and 10,000 validation structures, a 5–70 heavy-atom filter and 60–1,000 g/mol molecular-weight filter. Do not present its filtering criteria as those of the final full-ChEMBL corpus.

## PPI data and fine-tuning subsets

- **248 characterized PPI reference structures:** `PPI_molecule_summary_248.csv`. The separate activity-record file contains 309 entries; multiple observations can refer to the same molecule. Experimental results can depend on assay conditions.
- **802 omeprazole-related source entries:** `analysis/omeprazole_series/full_chembl37_corrected_matches.csv`. These arose from the corrected broad search and similarity selection, not exclusively from compounds with measured PPI activity.
- **786 Transformer target-fine-tuning structures:** obtained from those 802 candidates using the Transformer-side preparation.
- **771 S4-compatible structures:** 802 inputs minus 29 duplicates after representation normalization, one unsupported pretrained-S4 token and one sequence exceeding the checkpoint limit. **694 train / 77 validation**.
- **247 S4 reference-set structures:** 248 source rows minus one duplicate after standardization. **222 train / 25 validation**.

The `s4_ppi786` directory name refers to the common source experiment, not 786 accepted S4 molecules.

## Representation and tokenization differences

The in-house Transformer is a **character-level SMILES language model**. The S4 implementation uses chemical token segmentation plus a **fixed checkpoint vocabulary** (37 tokens in the inspected checkpoint). The S4 preprocessing script validates that each molecule can be completely tokenized without introducing unseen token IDs, and uses the checkpoint's `sequence_length=99`. By default it converts the charge-separated sulfoxide depiction `[S+]([O-])` to `S(=O)` and removes stereochemistry. These choices can merge entries that were distinct in the Transformer dataset.

A tokenizer's regex supporting a chemical symbol does **not** imply that the saved S4 checkpoint has a learned embedding for that symbol. Likewise, the curated corpus's 11-element restriction is an upstream data-selection policy, not merely a tokenizer limitation.

## Avoiding leakage and overstatement

1. The full-ChEMBL preparation excludes *exact standardized* reference PPI structures; analogous scaffolds may still appear. This is not scaffold-disjoint splitting.
2. The 786/771 series is derived from ChEMBL 37 and may overlap molecules in the in-house pretrained corpus. The original S4 ChEMBL v31 corpus may also overlap target/reference compounds; it has not been comprehensively audited.
3. The model predicts chemical sequences and, separately, an exploratory pIC50 value. None of these output files establishes experimental potency or selectivity.
4. Distinguish **broad sulfoxide–benzimidazole motif matching** from **strict omeprazole-like pyridylmethyl–sulfoxide–benzimidazole connectivity**.

For commands and exact paths, see [`commands.md`](commands.md).
