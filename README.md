# PPI Deep Learning Drug Design

**Molecular generation with a causal Transformer and a Structured State Space (S4) model, supported by GINE-based exploratory activity prediction**

This repository studies deep-learning workflows for generating and evaluating small molecules related to **proton pump inhibitors (PPIs)**. The main contribution is a SMILES-based **autoregressive Transformer** pretrained on ChEMBL and fine-tuned on PPI-related structures. It is compared with an existing, ChEMBL-pretrained **S4 state-space molecular generator**. A **Graph Isomorphism Network with Edge Features (GINE)** provides supplementary, exploratory pIC₅₀ predictions.

> **Scientific scope:** Valid SMILES, scaffold similarity, drug-likeness metrics and predicted pIC₅₀ are not evidence of experimentally confirmed proton-pump inhibition. Generated compounds in this project have not been experimentally validated.

## Research questions

1. Can a causal Transformer learn to generate valid, structurally diverse SMILES after ChEMBL pretraining and PPI-specific fine-tuning?
2. How does the generation workflow compare with a published S4 state-space model?
3. How does sampling temperature affect validity and uniqueness?
4. How many generated structures retain a broad or strict PPI-related motif, and how novel are they relative to reference molecules?
5. Can a GINE activity predictor be used as an **exploratory** filter, with appropriate applicability-domain checks?

## Repository organization

The uploaded project includes the following principal paths. Some scripts also occur as legacy copies at repository root; the subfolders are the intended locations for the organized project.

```text
PPI-Deep-Learning-Drug-Design/
├── README.md
├── data_preparation/
│   ├── 01_ChEMBL_Data_Preparation.ipynb
│   ├── 10_prepare_full_chembl.py
│   └── data_audit.py
├── gnn/
│   ├── 11_GINE_Full_ChEMBL_Pretraining.py
│   └── 12_PPI_Full_ChEMBL_FineTuning.py
├── transformer/
│   ├── 13_PPI_Molecular_Transformer_Pretraining.py
│   ├── 14_Transformer_Generation_Evaluation.py
│   ├── 15_PPI_Transformer_FineTuning.py
│   └── 16_Extract_Unique_PPI_Core.py
├── s4_fine_tuning/
│   ├── S4_01_PPI_Data_Preparation.py
│   ├── S4_02_PPI_FineTuning.py
│   └── test-S4-kj.ipynb
├── evaluation/
│   └── 18_PPI_Preliminary_Cheminformatics.py
├── 04_PPI_Model_Comparison.ipynb
├── 05_PPI_screening_diagnostics.py
├── 06_PPI_Applicability_Domain.py
├── 07_PPI_omeprazole_scaffold_search.py
├── 08_PPI_corrected_scaffold_search.py
├── PPI_molecule_summary_248.csv
└── analysis/ ...
```

**Note:** The existing `04_PPI_Model_Comparison.ipynb` is used for the common model-comparison workflow. Separate `evaluate_generated_molecules.py` or `predict_generated_pIC50.py` scripts are not prerequisites for this repository. The graph-diffusion prototype (`17_PPI_Graph_Diffusion.py`) is outside the scope of this release.

## Dataset and molecular representation

### ChEMBL pretraining

The custom Transformer and GINE workflows use a curated **ChEMBL 37** molecular corpus. The full-ChEMBL preparation script validates SMILES using RDKit, selects the largest organic fragment, retains the elements **H, B, C, N, O, F, P, S, Cl, Br and I**, and filters for **5–80 heavy atoms** and **SMILES length ≤ 200**. Canonical isomeric SMILES preserve the formal charges and stereochemistry represented by the source molecules. Exact duplicates and a designated reference PPI set are excluded; the remaining molecules are allocated to deterministic train/validation/test splits.

| Partition | Structures |
|---|---:|
| Training | 2,462,166 |
| Validation | 137,158 |
| Test | 136,279 |
| **Total** | **2,735,603** |

### PPI-related fine-tuning

- **248 reference PPI structures:** experimentally characterized reference collection, associated with multiple activity records.
- **786 Transformer fine-tuning structures:** retained from an initial collection of 802 ChEMBL PPI-related candidates.
- **771 S4-compatible structures:** retained from the same initial collection after S4-specific preprocessing; split into **694 training / 77 validation**.
- **247 S4-compatible reference structures:** derived from the 248-structure collection after deduplication; split into **222 training / 25 validation**.

The S4 preprocessing checks token compatibility with the **actual saved 37-token checkpoint vocabulary**, removes stereochemistry by default, and converts the specific charge-separated sulfoxide representation `[S+]([O-])` to `S(=O)`. These choices should be acknowledged when comparing the models: the tokenizer's *possible* chemical symbols are not the same as a pretrained checkpoint's *learned* token vocabulary.

**Representations:** The Transformer operates on character-level SMILES tokens; S4 uses chemistry-oriented SMILES segmentation. GINE operates on **molecular graphs** (atoms as nodes, bonds as edges), not protein–ligand complex graphs.

## Models

### Causal Transformer — primary model

A PyTorch model using causal masking with `TransformerEncoderLayer` modules (autoregressive next-token modeling).

| Setting | Value |
|---|---|
| Transformer blocks | 6 |
| Attention heads | 8 |
| Token embedding dimension | 256 |
| Feedforward dimension | 1,024 |
| Dropout | 0.1 |
| Maximum sequence length | 202 |
| Approximate parameter count | 4.81 million |
| ChEMBL pretraining | 15 epochs |
| PPI-specific fine-tuning | 30 epochs |

The class name `TransformerEncoderLayer` reflects the PyTorch implementation; causal masking makes the task a **decoder-style autoregressive language model**.

### S4 — comparison model

The S4 implementation is sourced from the authors' [S4 for de novo drug design](https://github.com/molML/s4-for-de-novo-drug-design) repository. It uses **structured state-space sequence layers**, not Transformer self-attention. The official ChEMBL-v31 checkpoint inspected for this project contains **3,310,663 parameters**, a vocabulary of **37 tokens**, and **256-dimensional embeddings**. The S4 model was **not pretrained from scratch here**; the contribution is PPI-specific preparation, fine-tuning, sampling and analysis. Two independent fine-tuning datasets were studied (771 and 247 usable molecules).

### GINE — supplementary activity model

GINE uses atom/bond graph features for molecular-property learning. The PPI-specific model provides **exploratory predicted pIC₅₀**, where

\[
\mathrm{pIC}_{50}=-\log_{10}(\mathrm{IC}_{50}\,[\mathrm{mol/L}]).
\]

A model score is not a measured IC₅₀. Predictions outside the model's training chemical domain must be interpreted cautiously.

## Previously recorded generation results

The following are **recorded experiment summaries**, not new calculations from this README.

| Experiment | Generated | Valid | Unique valid |
|---|---:|---:|---:|
| Fine-tuned Transformer (786) | 1,000 | 988 | 820 |
| Fine-tuned S4 (771), T=1.00 | 1,000 | 959 | 952 |
| Fine-tuned S4 (247), T=1.00 | 1,000 | 957 | 957 |

Of the Transformer's **820 unique valid** structures, **116** matched a **broad sulfoxide–benzimidazole motif**, spanning **48 Murcko scaffolds**. These 116 should **not** automatically be described as strict classical PPIs: a full pyridylmethyl-sulfoxide-benzimidazole connectivity test and biological validation are distinct requirements.

### S4 sampling temperature

All three runs used the **same S4-771 checkpoint**, with 1,000 generation attempts per setting.

| Temperature | Valid | Unique valid | Validity |
|---:|---:|---:|---:|
| 0.70 | 989 | 928 | 98.9% |
| 0.85 | 984 | 963 | 98.4% |
| 1.00 | 959 | 952 | 95.9% |

T=0.85 produced the largest **absolute number of unique valid molecules** in these runs; this does not establish the best temperature for target-relevant chemical scaffolds or bioactivity. The two generative models differ in tokenizer, pretraining corpus and sampling implementation, so the comparison is not an isolated architecture-only benchmark.

## Installation

Use separate environments for the independently implemented GINE/Transformer workflow and the S4 source implementation. The original experiments used Conda environments named `ppi_gnn` and `s4dd_torch2`. Exact versions depend on the available GPU and CUDA stack.

```bash
# For the custom Transformer, GINE and RDKit analysis
conda activate ppi_gnn

# For S4 experiments
conda activate s4dd_torch2
```

The upstream S4 package can be installed from its official repository:

```bash
git clone https://github.com/molML/s4-for-de-novo-drug-design.git
cd s4-for-de-novo-drug-design
python -m pip install -e .
```

Do not commit the authors' complete upstream repository or large pretrained checkpoint inside this project's source tree.

## Reproducing selected experiments

The paths below are relative to this project's root **after** its directory layout is finalized. They are representative commands for scripts whose interfaces have been checked. Source data and official S4 checkpoint paths must be supplied locally. Some existing scripts contain paths from the original working directory and may require adjustment after reorganization.

### 1. Download ChEMBL 37 chemical representations

```bash
mkdir -p data/raw
wget -c \
  https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/chembl_37_chemreps.txt.gz \
  -O data/raw/chembl_37_chemreps.txt.gz
```

### 2. Prepare the ChEMBL dataset

```bash
python data_preparation/10_prepare_full_chembl.py \
  --input data/raw/chembl_37_chemreps.txt.gz \
  --ppi PPI_molecule_summary_248.csv \
  --out data/full_chembl37
```

### 3. Prepare an S4-specific PPI fine-tuning corpus

```bash
python s4_fine_tuning/S4_01_PPI_Data_Preparation.py \
  --input analysis/omeprazole_series/full_chembl37_corrected_matches.csv \
  --checkpoint /path/to/official/chemblv31 \
  --out experiments/s4_ppi786/data \
  --seed 42 --val-frac 0.10
```

### 4. Fine-tune S4

```bash
python s4_fine_tuning/S4_02_PPI_FineTuning.py \
  --mode train \
  --checkpoint /path/to/official/chemblv31 \
  --data experiments/s4_ppi786/data \
  --out experiments/s4_ppi786 \
  --epochs 15 --batch-size 16 --lr 0.00002 --seed 42
```

### 5. Sample S4 at different temperatures

```bash
for temp in 0.70 0.85 1.00; do
  python s4_fine_tuning/S4_02_PPI_FineTuning.py \
    --mode sample \
    --checkpoint experiments/s4_ppi786/finetuned_model \
    --out "experiments/s4_temp_${temp}" \
    --num-samples 1000 --sample-batch-size 32 \
    --temperature "$temp"
done
```

### 6. Evaluate and compare models

Use the existing `04_PPI_Model_Comparison.ipynb` along with the analysis scripts and the generated molecular CSV files. The exact commands for Transformer pretraining/fine-tuning and GINE training should be taken from each script's documented CLI rather than guessed here.

## Evaluation measures

- SMILES/RDKit **validity** and canonical **uniqueness**.
- Broad and strict **PPI-related substructure** retention.
- Exact overlap with fine-tuning and ChEMBL reference sets (**novelty**).
- Morgan fingerprint **Tanimoto similarity** to known molecules.
- **Murcko scaffold** counts and scaffold diversity.
- RDKit molecular descriptors (MW, LogP, TPSA, HBD, HBA, rotatable bonds and QED).
- Optional, supplementary **GINE predicted pIC₅₀**, with applicability-domain analysis.

## Limitations and responsible interpretation

This is a **computational study**, not an experimental drug-discovery validation. Limitations include small target-specific datasets, differences between ChEMBL versions and tokenizers, possible overlap with the original S4 pretraining corpus, differences in stereochemical/charge handling, and imperfect activity-prediction generalization. Reported generation validity is insufficient to claim target-specific inhibitory activity.

## References and acknowledgments

- [ChEMBL](https://www.ebi.ac.uk/chembl/) — molecular structures and activity data.
- [RDKit](https://www.rdkit.org/) — molecular processing, substructure matching and descriptors.
- [PyTorch](https://pytorch.org/) — neural-network implementation.
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) — graph neural network tools.
- [S4 for de novo drug design](https://github.com/molML/s4-for-de-novo-drug-design) — original S4 code and checkpoint; cite its associated publication when using the model.

**License:** Add an explicit license after checking compatibility with reused code and the official S4 dependency. No license is implied by this README.
