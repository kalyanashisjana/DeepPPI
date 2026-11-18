#!/usr/bin/env python3
"""Predict exploratory PPI pIC50/IC50 from generated-molecule CSV using fine-tuned GINE.

  python evaluation/19_PPI_GINE_Predict_Generated.py \
      --input example/unique_core_molecules/PPI_116_cheminformatics.csv \
      --checkpoint checkpoints/ppi_full_finetuning/Scaffold_FullChEMBL.pt \
      --output evaluation/PPI_116_GINE_predictions.csv

Or set --input to any CSV with a canonical_smiles or smiles column.

Requires original scripts gnn/11_GINE_Full_ChEMBL_Pretraining.py and
    gnn/12_PPI_Full_ChEMBL_FineTuning.py from THIS project.

IMPORTANT: The checkpoint MUST be the final pIC50 fine-tuned regressor from
12_PPI_Full_ChEMBL_FineTuning.py. Do not substitute its masked-atom pretrained
encoder checkpoint: that checkpoint cannot predict pIC50.

The exported values are MODEL PREDICTIONS, not measured IC50, validated
proton-pump inhibition, binding affinity or clinical efficacy.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader


def import_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def locate_project_root():
    for base in (Path.cwd().resolve(), Path(__file__).resolve().parent):
        for parent in (base, *base.parents):
            if (parent / 'gnn' / '11_GINE_Full_ChEMBL_Pretraining.py').exists() and (
                parent / 'gnn' / '12_PPI_Full_ChEMBL_FineTuning.py'
            ).exists():
                return parent
    raise FileNotFoundError(
        'Cannot find project root with the original gnn/11_... and gnn/12_... scripts.'
    )


def run(args):
    root = locate_project_root()
    input_path = Path(args.input).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f'Input CSV not found: {input_path}')
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f'Fine-tuned model checkpoint not found: {checkpoint_path}\n'
            'Look for checkpoints/ppi_full_finetuning/Scaffold_FullChEMBL.pt '
            'in your ORIGINAL working folder. The checkpoint may have been excluded '
            'from GitHub by .gitignore. Without the actual fine-tuned weights, '
            'the pIC50 values cannot be calculated.'
        )

    encoder = import_script(root / 'gnn' / '11_GINE_Full_ChEMBL_Pretraining.py', 'deep_ppi_gine_encoder')
    fine_tune = import_script(root / 'gnn' / '12_PPI_Full_ChEMBL_FineTuning.py', 'deep_ppi_gine_finetune')

    device = torch.device('cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
    if not isinstance(ckpt, dict) or not {'model_state_dict', 'target_mean', 'target_std'} <= ckpt.keys():
        raise ValueError(
            'Not a 12_PPI_Full_ChEMBL_FineTuning.py regression checkpoint: '
            'expected model_state_dict, target_mean, target_std. '
            'Do not use the 11_... masked-atom pretraining checkpoint.'
        )
    if 'encoder_elements' in ckpt and list(ckpt['encoder_elements']) != list(encoder.ELEMENTS):
        raise ValueError('Checkpoint atom-element vocabulary differs from current featurizer.')

    model = fine_tune.PPIRegressor(encoder.MolecularGINEEncoder).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    target_mean = float(ckpt['target_mean'])
    target_std = float(ckpt['target_std'])
    if not np.isfinite(target_mean) or not np.isfinite(target_std) or target_std <= 0:
        raise ValueError('Checkpoint target normalization metadata is invalid.')

    df = pd.read_csv(input_path)
    candidate_columns = [args.smiles_column] if args.smiles_column else [
        'canonical_smiles_checked', 'canonical_smiles', 'smiles', 'raw_smiles'
    ]
    smiles_column = next((c for c in candidate_columns if c in df.columns), None)
    if smiles_column is None:
        raise ValueError('CSV must have one of: canonical_smiles_checked, canonical_smiles, smiles, raw_smiles '
                         'or specify --smiles-column. Found: ' + ', '.join(df.columns))

    if 'molecule_id' not in df.columns:
        df.insert(0, 'molecule_id', [f'GEN_{i:04d}' for i in range(1, len(df) + 1)])

    good_indices, graphs = [], []
    statuses = []
    for idx, smiles in enumerate(df[smiles_column].tolist()):
        if not isinstance(smiles, str) or not smiles.strip():
            statuses.append('empty_smiles')
            continue
        try:
            graph = encoder.smiles_to_graph(smiles)
        except Exception:
            graph = None
        if graph is None:
            statuses.append('invalid_smiles')
            continue
        statuses.append('ok')
        good_indices.append(idx)
        graphs.append(graph)

    df['gine_status'] = statuses
    df['predicted_pIC50'] = np.nan
    df['predicted_IC50_nM'] = np.nan
    df['predicted_IC50_uM'] = np.nan
    df['prediction_model'] = ckpt.get('metrics', {}).get('model', checkpoint_path.stem)
    df['prediction_split'] = ckpt.get('metrics', {}).get('split', 'unknown')

    predictions = []
    if graphs:
        loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)
        with torch.inference_mode():
            for batch in loader:
                batch = batch.to(device)
                values = model(batch).detach().cpu().numpy().reshape(-1)
                predictions.extend(values.tolist())
        predictions = np.asarray(predictions, dtype=np.float64) * target_std + target_mean
        # pIC50 = 9 - log10(IC50[nM])
        ic50_nM = np.power(10.0, 9.0 - predictions)
        df.loc[good_indices, 'predicted_pIC50'] = predictions
        df.loc[good_indices, 'predicted_IC50_nM'] = ic50_nM
        df.loc[good_indices, 'predicted_IC50_uM'] = ic50_nM / 1000.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f'Loaded: {len(df)} molecules; prediction succeeded: {len(good_indices)}; device: {device}')
    print('Checkpoint evaluation metrics:', json.dumps(ckpt.get('metrics', {}), default=str))
    if predictions.size:
        ranked = df[df['gine_status'] == 'ok'].sort_values('predicted_pIC50', ascending=False)
        cols = ['molecule_id', smiles_column, 'predicted_pIC50', 'predicted_IC50_nM']
        print('\nTop 10 by model-predicted pIC50 (EXPLORATORY, NOT MEASURED):')
        with pd.option_context('display.max_colwidth', 50, 'display.width', 160):
            print(ranked[cols].head(10).to_string(index=False, formatters={
                'predicted_pIC50': '{:.3f}'.format,
                'predicted_IC50_nM': '{:.3g}'.format,
            }))
        print('\nPrediction distribution (pIC50):')
        print(ranked['predicted_pIC50'].describe().round(3).to_string())
    print('\nSaved predictions:', output_path)
    print('WARNING: These are ligand-only model estimates for an assay-specific PPI endpoint, '
          'not experimentally measured IC50 or a validated binding-affinity score.')

    if args.plot and predictions.size:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(predictions, bins=18, edgecolor='white')
        ax.set_xlabel('GINE-predicted pIC50 (exploratory)')
        ax.set_ylabel('Number of generated molecules')
        ax.set_title('Predicted PPI activity of generated molecules')
        fig.tight_layout()
        plot_path = output_path.with_name(output_path.stem + '_histogram.png')
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        print('Saved histogram:', plot_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True, help='Input CSV of generated molecules')
    ap.add_argument('--checkpoint', required=True, help='Fine-tuned PPI GINE .pt regression checkpoint')
    ap.add_argument('--output', required=True, help='Output prediction CSV')
    ap.add_argument('--smiles-column', default=None, help='Override SMILES column autodetection')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--cpu', action='store_true', help='Force CPU inference')
    ap.add_argument('--plot', action='store_true', help='Save a histogram of predicted pIC50')
    args = ap.parse_args()
    if args.batch_size < 1:
        ap.error('--batch-size must be positive')
    run(args)


if __name__ == '__main__':
    main()
