# Reproducible commands: ChEMBL download to final evaluation

These commands were cross-checked against the **actual argument definitions in the uploaded Python scripts**, their source docstrings, and relevant saved manifests. They give a reproducible *workflow*, not a claim that every command was executed in precisely this order. Some earlier experiments were run from notebooks. Run the commands from the **project root** where paths `data/`, `analysis/`, `generation/`, and `checkpoints/` are located.

**Note:** Keep `gnn/11_GINE_Full_ChEMBL_Pretraining.py` and `gnn/12_PPI_Full_ChEMBL_FineTuning.py` in the same directory: fine-tuning imports the first module by file path. Keep `transformer/13_PPI_Molecular_Transformer_Pretraining.py` and `transformer/15_PPI_Transformer_FineTuning.py` together for the same reason. The uploaded archive has both subfolder and root-level duplicates; the commands below target the organized copies.

## A. Environment setup

```bash
# Example: existing environments (do not recreate if already configured)
conda activate ppi_gnn
python -c "import torch, rdkit, pandas; print(torch.__version__, rdkit.__version__)"

# For published S4, use separate environment
conda activate s4dd_torch2
# Install official S4 source in editable mode if it is not yet installed:
# git clone https://github.com/molML/s4-for-de-novo-drug-design.git
# cd s4-for-de-novo-drug-design && python -m pip install -e .
```

See `requirements/ppi_gnn_requirements.txt` and `requirements/s4_requirements.txt`. They are **dependency guides**, not a verified environment lockfile; make a full `pip freeze` in each working environment if exact reproduction is required.

## B. Download ChEMBL 37

```bash
mkdir -p data/raw
wget -c \
  https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/chembl_37_chemreps.txt.gz \
  -O data/raw/chembl_37_chemreps.txt.gz
```

The uploaded `data/processed/chembl37_pretrain_manifest.json` records SHA-256 `ea6181ce8dc7af41974e35b92e1febb0c9dcbe2c62f7ccc4a5d983ac19f696e7` for the originally downloaded gzip, which you may verify with `sha256sum data/raw/chembl_37_chemreps.txt.gz`.

## C. Prepare curated full ChEMBL corpus

```bash
python data_preparation/10_prepare_full_chembl.py \
  --input data/raw/chembl_37_chemreps.txt.gz \
  --ppi PPI_molecule_summary_248.csv \
  --out data/full_chembl37
```

Expected output: `chembl37_train.csv` (2,462,166), `chembl37_val.csv` (137,158), `chembl37_test.csv` (136,279), and manifest. A separate earlier 100k-corpus experiment is in `data_preparation/01_ChEMBL_Data_Preparation.ipynb`.

## D. Preliminary GINE masked-atom pretraining

```bash
python gnn/11_GINE_Full_ChEMBL_Pretraining.py \
  --train data/full_chembl37/chembl37_train.csv \
  --val data/full_chembl37/chembl37_val.csv \
  --epochs 20 --batch-size 128 --workers 2 --val-limit 10000

# Resume an interrupted run
python gnn/11_GINE_Full_ChEMBL_Pretraining.py \
  --resume checkpoints/full_chembl_gine/last.pt
```

## E. GINE PPI activity fine-tuning

```bash
python gnn/12_PPI_Full_ChEMBL_FineTuning.py \
  --data PPI_molecule_summary_248.csv \
  --full-checkpoint checkpoints/full_chembl_gine/best.pt \
  --out checkpoints/ppi_full_finetuning
```

Supplementary screening and applicability-domain evaluations were run from notebooks and these provided root-level scripts:

```bash
python 05_PPI_screening_diagnostics.py
python 06_PPI_Applicability_Domain.py
python 07_PPI_omeprazole_scaffold_search.py
python 08_PPI_corrected_scaffold_search.py
python 08_PPI_corrected_scaffold_search.py --full
```

**Caution:** Those four scripts use several fixed relative paths. If you have moved them into subfolders, preserve those inputs or adjust paths. The `--full` search reads the local ChEMBL gzip.

## F. Transformer pretraining (main model)

```bash
# Short smoke test
python transformer/13_PPI_Molecular_Transformer_Pretraining.py \
  --train-limit 10000 --val-limit 1000 --epochs 2 \
  --out checkpoints/smiles_transformer_smoke

# Full 15-epoch pretraining
python transformer/13_PPI_Molecular_Transformer_Pretraining.py \
  --epochs 15 --batch-size 64 --workers 2

# Resume
python transformer/13_PPI_Molecular_Transformer_Pretraining.py \
  --epochs 15 --batch-size 64 --workers 2 \
  --resume checkpoints/smiles_transformer/last.pt
```

## G. Unconditional Transformer generation and evaluation

```bash
python transformer/13_PPI_Molecular_Transformer_Pretraining.py \
  --mode sample --checkpoint checkpoints/smiles_transformer/best.pt \
  --num-samples 1000 --temperature 1.0 --top-k 30 \
  --output generation/transformer_unconditional.csv

python transformer/14_Transformer_Generation_Evaluation.py \
  --samples generation/transformer_unconditional.csv \
  --train data/full_chembl37/chembl37_train.csv \
  --ppi PPI_molecule_summary_248.csv \
  --out generation/transformer_evaluation.csv
```

## H. Find omeprazole-related fine-tuning structures

```bash
python 08_PPI_corrected_scaffold_search.py --full
# Confirm analysis/omeprazole_series/full_chembl37_corrected_matches.csv exists
```

The saved `full_chembl37_corrected_matches.csv` had 802 source rows. The corrected search supports multiple sulfoxide SMILES representations; it is not a set of 802 experimentally confirmed inhibitors.

## I. PPI Transformer preparation, fine-tuning and generation

```bash
python transformer/15_PPI_Transformer_FineTuning.py \
  --mode prepare \
  --series analysis/omeprazole_series/full_chembl37_corrected_matches.csv \
  --selection all

python transformer/15_PPI_Transformer_FineTuning.py \
  --mode train --epochs 30 --batch-size 16 --lr 2e-5 \
  --pretrained checkpoints/smiles_transformer/best.pt

python transformer/15_PPI_Transformer_FineTuning.py \
  --mode sample --num-samples 1000 \
  --temperature 0.90 --top-k 30 \
  --output generation/transformer_ppi_finetuned_1000.csv

python transformer/14_Transformer_Generation_Evaluation.py \
  --samples generation/transformer_ppi_finetuned_1000.csv \
  --out generation/transformer_ppi_finetuned_evaluated.csv

python transformer/16_Extract_Unique_PPI_Core.py

python evaluation/18_PPI_Preliminary_Cheminformatics.py \
  --input generation/unique_core_molecules/PPI_116_unique_core.csv \
  --ppi PPI_molecule_summary_248.csv \
  --chembl data/full_chembl37/chembl37_train.csv \
  --output generation/unique_core_molecules/PPI_116_cheminformatics.csv
```

**Important:** Script 16 reads fixed input `generation/transformer_ppi_finetuned_evaluated.csv`. Confirm the generated file exists before running it. The Transformer evaluated file includes `omeprazole_sulfoxide_benzimidazole_motif`, which is a **broad motif flag**, not strict classical PPI connectivity.

## J. Transformer sampling-temperature study

```bash
for temp in 0.70 0.85 1.00; do
  python transformer/15_PPI_Transformer_FineTuning.py \
    --mode sample --num-samples 1000 --temperature "$temp" --top-k 30 \
    --output "generation/transformer_temp_${temp}.csv"
done
```

All three output CSVs are included in the uploaded project archive. The script's default random seed is 42, so these three batches should not be presented as independent-seed replicates.

## K. S4 installation and official checkpoint

The S4 code is from https://github.com/molML/s4-for-de-novo-drug-design, not an in-house reimplementation. Obtain the published ChEMBL v31 model artifacts separately, for example from the paper's linked Zenodo deliverables. In the original local setup, the directory contained `model.pt` and `init_arguments.json` under `official_results/deliverables/designs-and-models/chemblv31`.

```bash
conda activate s4dd_torch2
# export S4_PRETRAINED=/absolute/path/to/official_results/deliverables/designs-and-models/chemblv31
```

## L. Prepare S4-771 (from 802 source rows)

```bash
python s4_fine_tuning/S4_01_PPI_Data_Preparation.py \
  --input analysis/omeprazole_series/full_chembl37_corrected_matches.csv \
  --checkpoint "$S4_PRETRAINED" \
  --out s4_fine_tuning/experiments/s4_ppi786/data \
  --seed 42 --val-frac 0.10

python s4_fine_tuning/S4_02_PPI_FineTuning.py \
  --mode train --checkpoint "$S4_PRETRAINED" \
  --data s4_fine_tuning/experiments/s4_ppi786/data \
  --out s4_fine_tuning/experiments/s4_ppi786 \
  --epochs 15 --batch-size 16 --lr 0.00002 --seed 42
```

## M. Prepare and train S4-247 (characterized reference set)

```bash
python s4_fine_tuning/S4_01_PPI_Data_Preparation.py \
  --input PPI_molecule_summary_248.csv \
  --checkpoint "$S4_PRETRAINED" \
  --out s4_fine_tuning/experiments/s4_ppi248/data \
  --seed 42 --val-frac 0.10

python s4_fine_tuning/S4_02_PPI_FineTuning.py \
  --mode train --checkpoint "$S4_PRETRAINED" \
  --data s4_fine_tuning/experiments/s4_ppi248/data \
  --out s4_fine_tuning/experiments/s4_ppi248 \
  --epochs 15 --batch-size 16 --lr 0.00002 --seed 42
```

## N. Sample S4-771 at three temperatures

```bash
for temp in 0.70 0.85 1.00; do
  python s4_fine_tuning/S4_02_PPI_FineTuning.py \
    --mode sample \
    --checkpoint s4_fine_tuning/experiments/s4_ppi786/finetuned_model \
    --out "s4_fine_tuning/experiments/s4_temp_${temp}" \
    --num-samples 1000 --sample-batch-size 32 \
    --temperature "$temp"
done
```

For S4-247 use checkpoint `s4_fine_tuning/experiments/s4_ppi248/finetuned_model` and a separate output directory. 

## O. Final comparison notebook and report

```bash
conda activate ppi_gnn
jupyter lab 04_PPI_Model_Comparison.ipynb
```

The uploaded notebook's code cells access files relative to the **repository root** (`data/`, `generation/`, `checkpoints/`, `analysis/`) and invoke the original screening scripts using `%run`. It is **not** a clean top-to-bottom executable pipeline: some cells rely on variables from earlier notebooks and there are sections from multiple experimental stages. Check inputs and execute only relevant sections. It contains graph-based prediction, screening diagnostics, molecular novelty/descriptor exploration, and checkpoint inspection.

There is **no need to invent** `evaluate_generated_molecules.py` or `predict_generated_pIC50.py` unless you independently refactor and validate those analyses from the notebook.

## P. Preserve environment details for exact reproduction

```bash
conda activate ppi_gnn
python -m pip freeze > requirements/ppi_gnn_environment_lock.txt

conda activate s4dd_torch2
python -m pip freeze > requirements/s4_environment_lock.txt
```

These commands must be run on the machine that has the environments. The included `requirements/*.txt` files document necessary packages but **cannot claim exact package pins from uninspected local installations**.
