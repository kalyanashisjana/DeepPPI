#!/usr/bin/env python3
"""Compare scratch vs full-ChEMBL pretrained GINE on identical PPI splits.

Run inside project/GNN:
 python 12_PPI_Full_ChEMBL_FineTuning.py
Optional 90k checkpoint:
 python 12_PPI_Full_ChEMBL_FineTuning.py --old-checkpoint checkpoints/gine_pretrained_full_best.pt

Uses the exact encoder and molecular featurizer defined by
11_GINE_Full_ChEMBL_Pretraining.py, loaded from the same directory.
"""
import argparse
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader


def get_encoder_code():
    script = Path(__file__).resolve().parent / '11_GINE_Full_ChEMBL_Pretraining.py'
    if not script.exists():
        raise FileNotFoundError(f'Need source GINE model script in same directory: {script}')
    spec = importlib.util.spec_from_file_location('ppi_full_pretrain_module', script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PPIRegressor(nn.Module):
    def __init__(self, encoder_cls, dropout=0.2):
        super().__init__()
        self.encoder = encoder_cls()
        self.head = nn.Sequential(nn.Linear(64,32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32,1))

    def forward(self, batch):
        z = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        return self.head(global_mean_pool(z, batch.batch)).squeeze(-1)


def load_pretained(model, checkpoint, device):
    checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if 'encoder_state_dict' not in checkpoint:
        raise KeyError('Checkpoint has no encoder_state_dict')
    model.encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=True)
    print('  Loaded pretrained encoder, epoch', checkpoint.get('epoch', 'unknown'))
    return model.to(device)


def build_splits(dataframe):
    # Identical split definitions to the original PPI_GINE_QSAR notebook.
    mols = [Chem.MolFromSmiles(smiles) for smiles in dataframe.canonical_smiles]
    assert all(m is not None for m in mols), 'Invalid PPI SMILES'
    scaffolds = np.asarray([MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols])
    idx = np.arange(len(dataframe))
    rpool, rtest = train_test_split(idx, test_size=0.20, random_state=42)
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    spool, stest = next(outer.split(idx, groups=scaffolds))
    rtrain, rval = train_test_split(rpool, test_size=0.20, random_state=17)
    inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=17)
    itr, iva = next(inner.split(spool, groups=scaffolds[spool]))
    splits = {'Random': (rtrain, rval, rtest), 'Scaffold': (spool[itr], spool[iva], stest)}
    for split, (train, val, test) in splits.items():
        if split == 'Scaffold':
            assert not (set(scaffolds[train]) & set(scaffolds[val]))
            assert not (set(scaffolds[train]) & set(scaffolds[test]))
            assert not (set(scaffolds[val]) & set(scaffolds[test]))
        print(f'{split}: train={len(train)}, val={len(val)}, test={len(test)}')
    return splits


def fit_one(label, data, train_idx, val_idx, test_idx, y, model_cls,
            module, checkpoint, device, cfg, output_dir):
    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    mean = float(np.mean(y[train_idx]))
    std = float(np.std(y[train_idx]))
    std = max(std, 1e-8)
    def make_loader(idx, shuffle):
        return DataLoader([data[int(i)] for i in idx], batch_size=cfg.batch_size, shuffle=shuffle)

    train_loader, val_loader, test_loader = (make_loader(indices, shuffle)
        for indices, shuffle in [(train_idx, True), (val_idx, False), (test_idx, False)])
    model = model_cls(module.MolecularGINEEncoder).to(device)
    if checkpoint is not None:
        load_pretained(model, checkpoint, device)
    # Entire encoder is fine-tuned, not frozen.
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best = float('inf')
    best_state = None
    best_epoch = -1
    patience_counter = 0
    history = []

    def evaluate(loader):
        model.eval()
        prediction, truth = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                p = model(batch).cpu().numpy() * std + mean
                prediction.extend(np.asarray(p).reshape(-1).tolist())
                truth.extend(np.asarray(batch.y.cpu().numpy()).reshape(-1).tolist())
        prediction = np.asarray(prediction)
        truth = np.asarray(truth)
        return mean_absolute_error(truth, prediction), prediction, truth

    for epoch in range(1, cfg.epochs+1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            target = (batch.y.view(-1) - mean) / std
            optimizer.zero_grad(set_to_none=True)
            preds = model(batch)
            loss = F.mse_loss(preds, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        val_mae, _, _ = evaluate(val_loader)
        history.append({'epoch': epoch, 'train_mse_norm': np.mean(losses), 'val_mae': val_mae})
        if val_mae < best - 1e-6:
            best, best_epoch, patience_counter = val_mae, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        if epoch == 1 or epoch % 20 == 0:
            print(f'  {label} epoch {epoch}: train MSE={np.mean(losses):.4f}; val MAE={val_mae:.4f}')
        if patience_counter >= cfg.patience:
            break

    if best_state is None:
        raise RuntimeError('No model selected')
    model.load_state_dict(best_state)
    test_mae, prediction, truth = evaluate(test_loader)
    metrics = {'split':label.split('/')[0], 'model':label.split('/')[1],
               'MAE':test_mae, 'RMSE': float(np.sqrt(mean_squared_error(truth, prediction))),
               'R2':float(r2_score(truth, prediction)), 'best_epoch':best_epoch,
               'val_MAE':best, 'train_n':len(train_idx), 'val_n':len(val_idx),
               'test_n':len(test_idx), 'seed':cfg.seed}
    run_name = label.replace('/', '_')
    torch.save({'model_state_dict':best_state, 'target_mean':mean, 'target_std':std,
                'metrics':metrics, 'encoder_elements':module.ELEMENTS,
                'featurizer':'11_GINE_Full_ChEMBL_Pretraining.py'}, output_dir / f'{run_name}.pt')
    pd.DataFrame(history).to_csv(output_dir / f'{run_name}_history.csv', index=False)
    pd.DataFrame({'chembl_id':[dataframe_ids[int(i)] for i in test_idx],
                  'true_pIC50':truth, 'pred_pIC50':prediction}).to_csv(
                    output_dir / f'{run_name}_predictions.csv', index=False)
    print(f'  {label}: test MAE={metrics["MAE"]:.3f} RMSE={metrics["RMSE"]:.3f} R2={metrics["R2"]:.3f}, best epoch={best_epoch}')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='PPI_molecule_summary_248.csv')
    parser.add_argument('--full-checkpoint', default='checkpoints/full_chembl_gine/best.pt')
    parser.add_argument('--old-checkpoint', default=None)
    parser.add_argument('--out', default='checkpoints/ppi_full_finetuning')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    cfg = parser.parse_args()

    module = get_encoder_code()
    datafile = Path(cfg.data)
    if not datafile.exists():
        raise FileNotFoundError(datafile)
    df = pd.read_csv(datafile).reset_index(drop=True)
    required = {'molecule_chembl_id', 'canonical_smiles', 'pIC50_median'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing columns: {required - set(df.columns)}')
    assert df.molecule_chembl_id.is_unique
    dataframe_ids = df.molecule_chembl_id.tolist()
    y = df.pIC50_median.to_numpy(dtype=np.float32)
    assert np.isfinite(y).all()
    graphs = []
    for smiles, value in zip(df.canonical_smiles, y):
        graph = module.smiles_to_graph(smiles)
        if graph is None:
            raise ValueError(f'Invalid molecular structure: {smiles}')
        graph.y = torch.tensor([float(value)], dtype=torch.float32)
        graphs.append(graph)
    splits = build_splits(df)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    models = [('Scratch', None), ('FullChEMBL', cfg.full_checkpoint)]
    if cfg.old_checkpoint:
        models.append(('Pretrain90k', cfg.old_checkpoint))
    rows = []
    for split_name, (train_idx, val_idx, test_idx) in splits.items():
        for model_name, ckpt in models:
            rows.append(fit_one(f'{split_name}/{model_name}', graphs, train_idx, val_idx,
                       test_idx, y, PPIRegressor, module, ckpt, device, cfg, out))
    result = pd.DataFrame(rows)
    result.to_csv(out / 'comparison.csv', index=False)
    print('\n', result.to_string(index=False))
    print('Saved:', out / 'comparison.csv')
