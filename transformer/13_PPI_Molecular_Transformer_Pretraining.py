#!/usr/bin/env python3
"""Autoregressive character-SMILES Transformer for full ChEMBL 37.

From project/GNN:
  python 13_PPI_Molecular_Transformer_Pretraining.py --train-limit 10000 --val-limit 1000 --epochs 2 --out checkpoints/smiles_transformer_smoke
  python 13_PPI_Molecular_Transformer_Pretraining.py --epochs 15 --batch-size 64 --workers 2
  python 13_PPI_Molecular_Transformer_Pretraining.py --epochs 15 --batch-size 64 --workers 2 --resume checkpoints/smiles_transformer/last.pt
  python 13_PPI_Molecular_Transformer_Pretraining.py --mode sample --checkpoint checkpoints/smiles_transformer/best.pt --num-samples 100 --output generation/transformer_unconditional.csv

This stage learns SMILES syntax/distribution, not PPI potency. No PPI labels or
filtered PPI series are needed to train it. Uses causal next-character prediction.
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

PAD, BOS, EOS, UNK = '<pad>', '<bos>', '<eos>', '<unk>'
SPECIAL = [PAD, BOS, EOS, UNK]


def build_vocab(path, limit=0):
    alphabet = set()
    with open(path, newline='', encoding='utf-8') as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if limit and i >= limit:
                break
            alphabet.update(row['canonical_smiles'])
    return SPECIAL + sorted(alphabet)


class StreamSMILES(IterableDataset):
    def __init__(self, path, stoi, max_len, limit=0, shuffle=False, seed=42, epoch=0, buffer=4096):
        self.path = str(path)
        self.stoi = stoi
        self.max_len = max_len
        self.limit = limit
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = epoch
        self.buffer = buffer

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        workers = worker.num_workers if worker else 1
        rng = random.Random(self.seed + self.epoch * 1009 + worker_id)
        memory = []
        with open(self.path, newline='', encoding='utf-8') as fh:
            for i, row in enumerate(csv.DictReader(fh)):
                if self.limit and i >= self.limit:
                    break
                if i % workers != worker_id:
                    continue
                smi = row['canonical_smiles']
                if not smi or len(smi) + 2 > self.max_len:
                    continue
                item = [self.stoi[BOS]] + [self.stoi.get(ch, self.stoi[UNK]) for ch in smi] + [self.stoi[EOS]]
                if not self.shuffle:
                    yield item
                elif len(memory) < self.buffer:
                    memory.append(item)
                else:
                    j = rng.randrange(len(memory))
                    yield memory[j]
                    memory[j] = item
        if self.shuffle:
            rng.shuffle(memory)
            yield from memory


def collate(rows):
    # Input x predicts y=x shifted by one. 0 is the padding token.
    width = max(len(x) - 1 for x in rows)
    x = torch.zeros((len(rows), width), dtype=torch.long)
    y = torch.zeros_like(x)
    for i, row in enumerate(rows):
        n = len(row) - 1
        x[i, :n] = torch.tensor(row[:-1])
        y[i, :n] = torch.tensor(row[1:])
    return x, y


class CausalSMILESTransformer(nn.Module):
    def __init__(self, vocab_size, max_len=202, d_model=256, layers=6, heads=8, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads,
                                           dim_feedforward=4*d_model,
                                           dropout=dropout, activation='gelu',
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        b, t = x.shape
        if t > self.max_len:
            raise ValueError(f'Sequence length {t} exceeds max_len {self.max_len}')
        z = self.dropout(self.tok(x) + self.pos(torch.arange(t, device=x.device))[None])
        causal_mask = torch.ones(t, t, device=x.device, dtype=torch.bool).triu(1)
        # Padding tokens are keys not to attend to.  All examples start with BOS.
        z = self.blocks(z, mask=causal_mask, src_key_padding_mask=(x == 0))
        return self.head(self.norm(z))


def loader(path, stoi, args, training, epoch=0):
    dataset = StreamSMILES(path, stoi, args.max_len,
                           args.train_limit if training else args.val_limit,
                           shuffle=training, seed=args.seed, epoch=epoch,
                           buffer=args.shuffle_buffer)
    return DataLoader(dataset, batch_size=args.batch_size,
                      num_workers=args.workers, collate_fn=collate,
                      pin_memory=torch.cuda.is_available())


def run_epoch(model, dl, device, optimizer=None, scaler=None, amp=False, grad_clip=1.0, log_every=1000):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    start = time.time()
    for step, (x, y) in enumerate(dl, 1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type='cuda', enabled=amp):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=0)
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
        valid = (y != 0)
        n = int(valid.sum().item())
        total_loss += loss.item() * n
        total_correct += ((logits.argmax(-1) == y) & valid).sum().item()
        total_tokens += n
        if training and log_every and step % log_every == 0:
            print(f'    step {step:,}: loss/token={total_loss/max(1,total_tokens):.4f}; elapsed={((time.time()-start)/60):.1f} min', flush=True)
    if not total_tokens:
        raise ValueError('No valid SMILES tokens found: inspect input file and max_len.')
    return total_loss / total_tokens, total_correct / total_tokens, total_tokens


def save_checkpoint(path, model, optimizer, scaler, scheduler, epoch, best, wait, vocab, config, hist):
    state = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                 scaler_state_dict=scaler.state_dict(), scheduler_state_dict=scheduler.state_dict(),
                 epoch=epoch, best_val_loss=best, wait=wait, vocab=vocab, config=config, history=hist,
                 torch_rng_state=torch.get_rng_state(), python_rng_state=random.getstate())
    if torch.cuda.is_available():
        state['cuda_rng_state'] = torch.cuda.get_rng_state_all()
    tmp = str(path) + '.tmp'
    torch.save(state, tmp)
    Path(tmp).replace(path)


def make_model(config, vocab_size, device):
    return CausalSMILESTransformer(vocab_size=vocab_size, **config).to(device)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        # Only load checkpoints created by this script / trusted files.
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        vocab, config = ckpt['vocab'], ckpt['config']
    else:
        print('Building character vocabulary from training CSV...', flush=True)
        vocab = build_vocab(args.train, args.train_limit)
        config = dict(max_len=args.max_len, d_model=args.d_model,
                      layers=args.layers, heads=args.heads, dropout=args.dropout)
    if args.d_model % args.heads != 0:
        raise ValueError('d_model must be divisible by heads')
    stoi = {ch: i for i, ch in enumerate(vocab)}
    (out / 'vocab.json').write_text(json.dumps(vocab, indent=2), encoding='utf-8')
    (out / 'model_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    model = make_model(config, len(vocab), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=args.lr_patience)
    amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    first, best, wait, history = 1, float('inf'), 0, []
    if args.resume:
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        first, best, wait, history = ckpt['epoch'] + 1, ckpt['best_val_loss'], ckpt['wait'], ckpt['history']
        torch.set_rng_state(ckpt['torch_rng_state'])
        random.setstate(ckpt['python_rng_state'])
        if device.type == 'cuda' and 'cuda_rng_state' in ckpt:
            torch.cuda.set_rng_state_all(ckpt['cuda_rng_state'])
    print(f'Device: {device}; vocabulary: {len(vocab)}; parameters: {sum(p.numel() for p in model.parameters()):,}', flush=True)
    print(f'Train: {args.train}; Validation: {args.val}; max_len={config["max_len"]}', flush=True)
    for epoch in range(first, args.epochs + 1):
        start = time.time()
        tr_loss, tr_acc, tr_tokens = run_epoch(model, loader(args.train, stoi, args, True, epoch), device,
                                              optimizer=optimizer, scaler=scaler, amp=amp, grad_clip=args.grad_clip,
                                              log_every=args.log_every)
        va_loss, va_acc, va_tokens = run_epoch(model, loader(args.val, stoi, args, False), device,
                                              amp=amp, log_every=0)
        scheduler.step(va_loss)
        improved = va_loss < best - args.min_delta
        if improved:
            best, wait = va_loss, 0
        else:
            wait += 1
        entry = dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, val_loss=va_loss,
                     val_acc=va_acc, train_tokens=tr_tokens, val_tokens=va_tokens,
                     lr=optimizer.param_groups[0]['lr'], minutes=(time.time()-start)/60)
        history.append(entry)
        save_checkpoint(out / 'last.pt', model, optimizer, scaler, scheduler, epoch, best, wait, vocab, config, history)
        if improved:
            save_checkpoint(out / 'best.pt', model, optimizer, scaler, scheduler, epoch, best, wait, vocab, config, history)
        with open(out / 'history.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(entry))
            writer.writeheader()
            writer.writerows(history)
        print(f'Epoch {epoch:02d} | train loss={tr_loss:.4f}, acc={tr_acc:.4f} | '
              f'val loss={va_loss:.4f}, acc={va_acc:.4f} | lr={entry["lr"]:.2e} | '
              f'{entry["minutes"]:.1f} min' + (' [BEST]' if improved else ''), flush=True)
        if wait >= args.patience:
            print(f'Early stopping at epoch {epoch}; best validation loss={best:.5f}', flush=True)
            break
    print(f'Best validation loss: {best:.6f}', flush=True)


@torch.no_grad()
def sample(args):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.error')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    vocab = ckpt['vocab']
    stoi = {x: i for i, x in enumerate(vocab)}
    model = make_model(ckpt['config'], len(vocab), device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    rows = []
    for i in range(args.num_samples):
        tokens = [stoi[BOS]]
        for _ in range(model.max_len - 1):
            x = torch.tensor(tokens, device=device, dtype=torch.long)[None]
            logits = model(x)[0, -1].float() / max(args.temperature, 1e-6)
            logits[stoi[PAD]] = -float('inf')
            logits[stoi[BOS]] = -float('inf')
            logits[stoi[UNK]] = -float('inf')
            if args.top_k > 0:
                k = min(args.top_k, logits.numel())
                values, _ = torch.topk(logits, k)
                logits[logits < values[-1]] = -float('inf')
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
            if nxt == stoi[EOS]:
                break
            tokens.append(nxt)
        raw = ''.join(vocab[t] for t in tokens[1:])
        mol = Chem.MolFromSmiles(raw)
        valid = mol is not None and len(raw) > 0
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True) if valid else ''
        novel_in_batch = bool(valid and canonical not in seen)
        if valid:
            seen.add(canonical)
        rows.append(dict(sample_id=i+1, raw_smiles=raw, canonical_smiles=canonical,
                         valid=valid, unique_in_batch=novel_in_batch))
    with open(output, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ['sample_id','raw_smiles','canonical_smiles','valid','unique_in_batch'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} samples to {output}; valid={sum(r["valid"] for r in rows)}; '
          f'unique valid={sum(r["unique_in_batch"] for r in rows)}', flush=True)
    print('Note: unconditional generation; no claims about PPI activity or novelty vs ChEMBL.', flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['train', 'sample'], default='train')
    p.add_argument('--train', default='data/full_chembl37/chembl37_train.csv')
    p.add_argument('--val', default='data/full_chembl37/chembl37_val.csv')
    p.add_argument('--out', default='checkpoints/smiles_transformer')
    p.add_argument('--resume', default='')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--train-limit', type=int, default=0)
    p.add_argument('--val-limit', type=int, default=10000)
    p.add_argument('--max-len', type=int, default=202)
    p.add_argument('--d-model', type=int, default=256)
    p.add_argument('--layers', type=int, default=6)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--lr-patience', type=int, default=2)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--min-delta', type=float, default=1e-4)
    p.add_argument('--shuffle-buffer', type=int, default=4096)
    p.add_argument('--log-every', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint', default='checkpoints/smiles_transformer/best.pt')
    p.add_argument('--num-samples', type=int, default=100)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top-k', type=int, default=30)
    p.add_argument('--output', default='generation/transformer_unconditional.csv')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    sample(args) if args.mode == 'sample' else train(args)
