# Model training and inference

## Project roles

The **causal SMILES Transformer** is the main deep-learning contribution. The published **S4 structured state-space model** provides an alternative generative architecture; the **GINE molecular graph model** serves as a supplementary exploratory pIC50 predictor. GINE is not a protein–ligand docking model: its input graph represents a ligand's atoms and bonds, not a protein–ligand complex.

The original uploaded archive contains the associated code both at repository root and in several named subfolders. The commands below assume execution **from the repository root** and retain runtime inputs under `data/`, `generation/`, `checkpoints/`, `analysis/` and `s4_fine_tuning/experiments/`.

## 1. GINE: graph feature learning and pIC50 regression

Files: `gnn/11_GINE_Full_ChEMBL_Pretraining.py`, `gnn/12_PPI_Full_ChEMBL_FineTuning.py`; earlier notebooks `02_GINE_Pretraining.ipynb`, `03_PPI_FineTuning.ipynb` and `PPI_GINE_QSAR.ipynb`.

Atoms become graph vertices, and chemical bonds become edges with chemical features. The first script trains a GINE encoder using a **masked-atom task** on the full curated ChEMBL corpus; the second compares PPI regression fine-tuning initialized from pretrained and scratch weights, and uses the 248-reference dataset. A molecular-level prediction is not an experimentally measured IC50.

```bash
python gnn/11_GINE_Full_ChEMBL_Pretraining.py \
  --train data/full_chembl37/chembl37_train.csv \
  --val data/full_chembl37/chembl37_val.csv \
  --epochs 20 --batch-size 128 --workers 2 --val-limit 10000

python gnn/12_PPI_Full_ChEMBL_FineTuning.py \
  --data PPI_molecule_summary_248.csv \
  --full-checkpoint checkpoints/full_chembl_gine/best.pt \
  --out checkpoints/ppi_full_finetuning
```

**Path caveat:** The GINE fine-tuning script dynamically loads the encoder definition from `11_GINE_Full_ChEMBL_Pretraining.py`. Both files must remain together in `gnn/`; run from the project root with valid input and checkpoint paths. Model prediction on generated SMILES is explored in `04_PPI_Model_Comparison.ipynb` and was not verified as a standalone `predict_generated_pIC50.py` program.

## 2. Custom causal Transformer

File: `transformer/13_PPI_Molecular_Transformer_Pretraining.py`.

Architecture: 6 `nn.TransformerEncoderLayer` blocks **with a causal mask** (therefore a decoder-style autoregressive language model), embedding width 256, 8 attention heads, feedforward width 1024, dropout 0.1, maximum sequence length 202, GELU, learnable token and position embeddings, and a next-character training objective. Checkpoint weight count is approximately 4.81 million. Each block uses standard Query, Key, Value and output projections; they are not the state-space A/B/C/D operators.

```bash
# Small installation/smoke test
python transformer/13_PPI_Molecular_Transformer_Pretraining.py \
  --train-limit 10000 --val-limit 1000 --epochs 2 \
  --out checkpoints/smiles_transformer_smoke

# Full training
python transformer/13_PPI_Molecular_Transformer_Pretraining.py \
  --epochs 15 --batch-size 64 --workers 2

# Resume if necessary
python transformer/13_PPI_Molecular_Transformer_Pretraining.py \
  --epochs 15 --batch-size 64 --workers 2 \
  --resume checkpoints/smiles_transformer/last.pt

# Unconditional sample
python transformer/13_PPI_Molecular_Transformer_Pretraining.py \
  --mode sample --checkpoint checkpoints/smiles_transformer/best.pt \
  --num-samples 1000 --output generation/transformer_unconditional.csv
```

File `transformer/15_PPI_Transformer_FineTuning.py` supports `--mode prepare`, `train` and `sample`. It selects the corrected 802-record series, prepares compatible sequences (786 in the documented run), trains for 30 epochs using the full-ChEMBL Transformer checkpoint, and samples autoregressively with temperature and top-k.

```bash
python transformer/15_PPI_Transformer_FineTuning.py --mode prepare
python transformer/15_PPI_Transformer_FineTuning.py --mode train --epochs 30
python transformer/15_PPI_Transformer_FineTuning.py \
  --mode sample --num-samples 1000 --temperature 0.90 --top-k 30
```

Token-generation validity and novelty are evaluated with `transformer/14_Transformer_Generation_Evaluation.py`; the core subset is extracted by `transformer/16_Extract_Unique_PPI_Core.py`; more extensive analysis resides in `evaluation/18_PPI_Preliminary_Cheminformatics.py` and `04_PPI_Model_Comparison.ipynb`.

## 3. Published S4 checkpoint and PPI fine-tuning

Source repository: https://github.com/molML/s4-for-de-novo-drug-design

S4 uses a structured state-space kernel, with the continuous-time form `dh/dt = A h + B u` and `y = C h + D u`. It uses an efficient convolution-like evaluation in training and recurrent-state updates at generation. Unlike the Transformer, it does not use ordinary Query–Key self-attention. The inspected official checkpoint has **3,310,663 stored parameter elements**, 37 vocabulary embeddings and 256-dimensional token embeddings. The weight tensors expose a structured S4 parameterization (`kernel.C`, `kernel.B`, `kernel.P`, `inv_w_real`, `w_imag`, `log_dt`, `D`) rather than a single unrestricted dense `A` matrix.

The project adds **independent target fine-tuning** from the original pretrained checkpoint in two experiments: 771 usable omeprazole-related structures and 247 reference PPI structures. S4-specific preprocessing removes stereochemistry by default and normalizes the charged sulfoxide depiction. Both experiments were trained for 15 epochs at learning rate `2e-5`, batch size 16, seed 42.

```bash
# Replace /path/to/chemblv31 with the extracted OFFICIAL checkpoint directory
python s4_fine_tuning/S4_01_PPI_Data_Preparation.py \
  --input analysis/omeprazole_series/full_chembl37_corrected_matches.csv \
  --checkpoint /path/to/chemblv31 \
  --out s4_fine_tuning/experiments/s4_ppi786/data \
  --seed 42 --val-frac 0.10

python s4_fine_tuning/S4_02_PPI_FineTuning.py \
  --mode train --checkpoint /path/to/chemblv31 \
  --data s4_fine_tuning/experiments/s4_ppi786/data \
  --out s4_fine_tuning/experiments/s4_ppi786 \
  --epochs 15 --batch-size 16 --lr 0.00002 --seed 42
```

There is a separate 247-molecule preparation and fine-tune run using `--input PPI_molecule_summary_248.csv` and the `s4_ppi248` paths. For identical training budgets, both fine-tunings must begin from the same original ChEMBL-pretrained checkpoint, not from one another.

## 4. Hyperparameter comparison

S4-771 sampling at `T=0.70, 0.85, 1.00` was completed. Separate Transformer output CSVs at these temperatures are also present in the uploaded archive (`generation/transformer_temp_*.csv`). These experiments change **inference temperature**, not training weights. The Transformer keeps top-k 30 in the documented protocol. With multiple random seeds, report mean and standard deviation, not just one sample's percentage.

## 5. Reproducibility reminders

- Use the `ppi_gnn` Python environment for the in-house models and RDKit analysis; use `s4dd_torch2` with the official `s4dd` package for S4.
- Run scripts from repository root because most input/output paths are **relative to the current working directory**.
- The `model_training.md` examples follow actual `argparse` options inspected in the supplied scripts; earlier cells/notebooks can still contain absolute local paths. Review notebook cells before executing.
- Install `s4dd` separately from its upstream source; the lightweight S4 requirements file does not copy or redistribute the upstream model.
- Do not compare cross-model loss values naively; their tokenizations and datasets differ.
