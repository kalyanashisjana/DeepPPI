#!/usr/bin/env python3
"""Fine-tune the official ChEMBL-pretrained S4 checkpoint; optionally sample and evaluate.

Always initializes from --checkpoint, never from a previous PPI fine-tuning output.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from s4dd import S4forDenovoDesign

RDLogger.DisableLog('rdApp.error')


def check_model_dir(path):
    path = Path(path)
    if not (path/'model.pt').is_file() or not (path/'init_arguments.json').is_file():
        raise FileNotFoundError(f'Expected model.pt and init_arguments.json under {path}')
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['train', 'sample'], required=True)
    ap.add_argument('--checkpoint', required=True, help='Pretrained checkpoint directory (train) or fine-tuned model directory (sample)')
    ap.add_argument('--data', default='experiments/s4_ppi786/data')
    ap.add_argument('--out', default='experiments/s4_ppi786')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num-samples', type=int, default=1000)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--sample-batch-size', type=int, default=32)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    chk = check_model_dir(args.checkpoint)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = S4forDenovoDesign.from_file(str(chk))
    model.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device={model.device}, input checkpoint={chk.resolve()}')

    if args.mode == 'train':
        train_file = Path(args.data)/'train.txt'
        val_file = Path(args.data)/'valid.txt'
        for p in (train_file, val_file):
            if not p.is_file() or not p.read_text().strip():
                raise FileNotFoundError(f'Missing or empty SMILES input: {p}')
        if args.epochs < 1 or args.batch_size < 1:
            ap.error('--epochs and --batch-size must be >= 1')
        model.n_max_epochs = args.epochs
        model.batch_size = args.batch_size
        model.learning_rate = args.lr
        # Source train() creates an Adam optimizer with this LR. It uses this model's
        # S4 weights from the checkpoint, not random initialization.
        history = model.train(
            training_molecules_path=str(train_file),
            val_molecules_path=str(val_file),
        )
        pd.DataFrame(history).to_csv(out/'history.csv', index_label='epoch_index')
        model_dir = out/'finetuned_model'
        model.save(str(model_dir))
        run = vars(args).copy()
        run.update(checkpoint_resolved=str(chk.resolve()), model_dir=str(model_dir.resolve()),
                   train_count=len(train_file.read_text().splitlines()),
                   val_count=len(val_file.read_text().splitlines()))
        (out/'run_config.json').write_text(json.dumps(run, indent=2))
        print('Saved fine-tuned S4:', model_dir)
        print('Final history:', {k: (v[-1] if v else None) for k, v in history.items()})
        print('Note: train() saves final epoch weights. Examine history.csv before claiming the best epoch.')
    else:
        if args.num_samples < 1 or args.sample_batch_size < 1 or args.temperature <= 0:
            ap.error('Sampling parameters must be positive')
        smiles, loglikes = model.design_molecules(
            n_designs=args.num_samples,
            batch_size=args.sample_batch_size,
            temperature=args.temperature,
        )
        seen = set()
        records = []
        for i, (s, ll) in enumerate(zip(smiles, loglikes), start=1):
            mol = Chem.MolFromSmiles(s)
            canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else None
            unique = canon is not None and canon not in seen
            if canon:
                seen.add(canon)
            records.append(dict(sample_id=i, raw_smiles=s, canonical_smiles=canon,
                                valid=bool(mol), unique_valid=unique, mean_log_likelihood=float(ll)))
        df = pd.DataFrame(records)
        csv_path = out / f's4_generated_{args.num_samples}.csv'
        df.to_csv(csv_path, index=False)
        valid = int(df['valid'].sum())
        print(f'Saved: {csv_path}')
        print(f'Generated: {len(df)} | Valid: {valid} ({valid/len(df)*100:.2f}%) | Unique valid: {len(seen)}')
        print('This is SMILES validity only; it does not establish PPI activity or novelty.')

if __name__ == '__main__':
    main()
