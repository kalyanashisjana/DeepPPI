#!/usr/bin/env python3
"""Disk-streaming GINE masked-atom pretraining over curated ChEMBL37.

Example (from GNN directory):
python 11_GINE_Full_ChEMBL_Pretraining.py --train data/full_chembl37/chembl37_train.csv --val data/full_chembl37/chembl37_val.csv --epochs 20 --batch-size 128 --workers 2 --val-limit 10000
Resume: append --resume checkpoints/full_chembl_gine/last.pt
"""
import argparse
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger
from torch import nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv

# Must match Notebook 02 exactly for comparable encoder weights.
ELEMENTS = [5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
ATOM_TO_ID = {z: i + 1 for i, z in enumerate(ELEMENTS)}
UNK_ID = len(ELEMENTS) + 1
MASK_ID = len(ELEMENTS) + 2
VOCAB_SIZE = MASK_ID + 1
BOND_TYPES = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    atom_ids = [ATOM_TO_ID.get(a.GetAtomicNum(), UNK_ID) for a in mol.GetAtoms()]
    edges, features = [], []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = [float(bond.GetBondType() == t) for t in BOND_TYPES]
        edges.extend(((a, b), (b, a)))
        features.extend((feat, feat))
    edge_index = (torch.tensor(edges, dtype=torch.long).T.contiguous()
                  if edges else torch.empty((2, 0), dtype=torch.long))
    edge_attr = (torch.tensor(features, dtype=torch.float32)
                 if edges else torch.empty((0, 4), dtype=torch.float32))
    return Data(x=torch.tensor(atom_ids, dtype=torch.long), edge_index=edge_index, edge_attr=edge_attr)


class StreamingGraphs(IterableDataset):
    """Stream CSV rows; shard them between DataLoader workers without duplication.

    Uses a small reservoir-like shuffle buffer for training; no full in-memory graph cache.
    Dataset order changes by epoch; worker sharding is deterministic.
    """
    def __init__(self, path, epoch=0, seed=42, shuffle=False, buffer_size=2048, limit=0):
        super().__init__()
        self.path = str(path)
        self.epoch = epoch
        self.seed = seed
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.limit = limit

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        workers = worker.num_workers if worker is not None else 1
        rng = random.Random(self.seed + 100003 * self.epoch + worker_id)
        buffer = []
        with open(self.path, newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            for row_idx, row in enumerate(reader):
                if self.limit and row_idx >= self.limit:
                    break
                if row_idx % workers != worker_id:
                    continue
                graph = smiles_to_graph(row['canonical_smiles'])
                if graph is None:
                    continue
                if not self.shuffle:
                    yield graph
                    continue
                if len(buffer) < self.buffer_size:
                    buffer.append(graph)
                else:
                    index = rng.randrange(len(buffer))
                    yield buffer[index]
                    buffer[index] = graph
        if self.shuffle:
            rng.shuffle(buffer)
            yield from buffer


class MolecularGINEEncoder(nn.Module):
    def __init__(self, hidden=64, num_layers=3, dropout=0.1):
        super().__init__()
        self.atom_embedding = nn.Embedding(VOCAB_SIZE, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout
        for _ in range(num_layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp, edge_dim=4))
            self.norms.append(nn.LayerNorm(hidden))

    def forward(self, x, edge_index, edge_attr):
        h = self.atom_embedding(x)
        for conv, norm in zip(self.convs, self.norms):
            h = F.relu(norm(h + conv(h, edge_index, edge_attr)))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class MaskedAtomGINE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MolecularGINEEncoder()
        self.head = nn.Linear(64, VOCAB_SIZE)

    def forward(self, x, edge_index, edge_attr):
        return self.head(self.encoder(x, edge_index, edge_attr))


def run_epoch(model, loader, device, optimizer=None, fixed_mask_seed=2026):
    train = optimizer is not None
    model.train(train)
    # Validation mask is seeded once per epoch so it is comparable across epochs.
    generator = None
    if not train:
        generator = torch.Generator(device=device).manual_seed(fixed_mask_seed)
    total_loss, total_correct, masked_total, molecules = 0., 0, 0, 0
    begin = time.monotonic()
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        mask = torch.rand(batch.x.shape, device=device, generator=generator) < 0.15
        if not mask.any():
            mask[0] = True
        masked_x = batch.x.clone()
        masked_x[mask] = MASK_ID
        with torch.set_grad_enabled(train):
            logits = model(masked_x, batch.edge_index, batch.edge_attr)
            loss = F.cross_entropy(logits[mask], batch.x[mask])
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        n = int(mask.sum())
        total_loss += float(loss.item()) * n
        total_correct += int((logits[mask].argmax(1) == batch.x[mask]).sum().item())
        masked_total += n
        molecules += batch.num_graphs
    if masked_total == 0:
        raise RuntimeError('No graphs/atoms were read; verify paths and CSV columns')
    return {'loss': total_loss / masked_total,
            'accuracy': total_correct / masked_total,
            'molecules': molecules, 'minutes': (time.monotonic()-begin)/60}


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val, patience_wait, history):
    # Save via rename to avoid partially-written checkpoints if interrupted.
    temp = path.with_suffix('.tmp')
    torch.save({
        'model_state_dict': model.state_dict(),
        'encoder_state_dict': model.encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch, 'val_loss': best_val, 'wait': patience_wait,
        'history': history,
        'elements': ELEMENTS, 'mask_id': MASK_ID, 'vocab_size': VOCAB_SIZE,
        'torch_rng_state': torch.get_rng_state(),
    }, temp)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='data/full_chembl37/chembl37_train.csv')
    parser.add_argument('--val', default='data/full_chembl37/chembl37_val.csv')
    parser.add_argument('--out', default='checkpoints/full_chembl_gine')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--buffer-size', type=int, default=2048)
    parser.add_argument('--val-limit', type=int, default=10000,
                        help='Read first N validation rows each epoch; 0 means all')
    parser.add_argument('--train-limit', type=int, default=0,
                        help='Smoke test: limit training rows (0=entire dataset)')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    for name in (args.train, args.val):
        if not Path(name).is_file():
            parser.error(f'Input file does not exist: {name}')
    if args.workers < 0:
        parser.error('--workers must be >= 0')
    RDLogger.DisableLog('rdApp.warning')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    model = MaskedAtomGINE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-6)
    start_epoch, best_val, wait, history = 1, float('inf'), 0, []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Move optimizer moments to device after loading CPU checkpoint.
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = int(checkpoint['epoch']) + 1
        best_val = float(checkpoint['val_loss'])
        wait = int(checkpoint.get('wait', 0))
        history = list(checkpoint.get('history', []))
        if 'torch_rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['torch_rng_state'])
        print(f'Resumed after epoch {start_epoch-1}', flush=True)

    print(f'Device: {device}; train={args.train}; val={args.val}', flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        train_ds = StreamingGraphs(args.train, epoch=epoch, seed=42, shuffle=True,
                                   buffer_size=args.buffer_size, limit=args.train_limit)
        val_ds = StreamingGraphs(args.val, shuffle=False, limit=args.val_limit)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.workers,
                                  pin_memory=device.type=='cuda')
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers,
                                pin_memory=device.type=='cuda')
        tr = run_epoch(model, train_loader, device, optimizer=optimizer)
        va = run_epoch(model, val_loader, device, optimizer=None)
        scheduler.step(va['loss'])
        improved = va['loss'] < best_val - 1e-6
        if improved:
            best_val, wait = va['loss'], 0
        else:
            wait += 1
        history.append({'epoch': epoch, 'train_loss': tr['loss'], 'val_loss': va['loss'],
                        'train_accuracy': tr['accuracy'], 'val_accuracy': va['accuracy'],
                        'train_molecules': tr['molecules'], 'val_molecules': va['molecules'],
                        'train_minutes': tr['minutes'], 'val_minutes': va['minutes'],
                        'lr': optimizer.param_groups[0]['lr']})
        print(f'Epoch {epoch:02d} | train loss {tr["loss"]:.4f}, acc {tr["accuracy"]:.4f} '
              f'({tr["molecules"]:,} molecules, {tr["minutes"]:.1f} min) | '
              f'val loss {va["loss"]:.4f}, acc {va["accuracy"]:.4f} '
              f'({va["molecules"]:,} molecules) | lr={optimizer.param_groups[0]["lr"]:.2e}', flush=True)
        # Always write resumable checkpoint, even if validation doesn't improve.
        save_checkpoint(output/'last.pt', model, optimizer, scheduler, epoch, best_val, wait, history)
        if improved:
            save_checkpoint(output/'best.pt', model, optimizer, scheduler, epoch, best_val, wait, history)
        import pandas as pd
        pd.DataFrame(history).to_csv(output/'history.csv', index=False)
        if wait >= args.patience:
            print(f'Early stopping after epoch {epoch}', flush=True)
            break
    print(f'Best validation loss: {best_val:.6f}', flush=True)


if __name__ == '__main__':
    main()
