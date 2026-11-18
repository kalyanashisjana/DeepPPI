#!/usr/bin/env python3
"""Fine-tune pretrained character-SMILES Transformer on an omeprazole-like series.

Run inside project/GNN alongside script 13:
  python 15_PPI_Transformer_FineTuning.py --mode prepare
  python 15_PPI_Transformer_FineTuning.py --mode train --epochs 30
  python 15_PPI_Transformer_FineTuning.py --mode sample --num-samples 1000

Default selection: all similarity-search matches, including core-retaining and
similarity-only analogues. Option --selection core restricts to the exact motif.
No experimental PPI activity is inferred from similarity or generation.
"""
import argparse
import csv
import importlib.util
import json
import random
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
from torch.utils.data import DataLoader

RDLogger.DisableLog('rdApp.error')

HERE = Path(__file__).resolve().parent
SCRIPT13 = HERE / '13_PPI_Molecular_Transformer_Pretraining.py'
if not SCRIPT13.exists():
    raise FileNotFoundError(f'Keep script 13 beside script 15: {SCRIPT13}')
spec = importlib.util.spec_from_file_location('smiles_pretrain', SCRIPT13)
pre = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pre)


def load_checkpoint(path):
    # Load only checkpoints generated locally by script 13 or 15.
    return torch.load(path, map_location='cpu', weights_only=False)


def motif(mol):
    queries = [
        '[#6]-[S;X3](=[O])-[c]1nc2ccccc2[nH]1',
        '[#6]-[S+;X3]([O-])-[c]1nc2ccccc2[nH]1',
    ]
    return any(mol.HasSubstructMatch(Chem.MolFromSmarts(q)) for q in queries)


def prepare(args):
    src = pd.read_csv(args.series)
    if 'canonical_smiles' not in src.columns:
        raise ValueError('Input series requires canonical_smiles column')
    if args.selection == 'core':
        if 'retains_sulfoxide_benzimidazole' not in src.columns:
            raise ValueError('Core selection requires retains_sulfoxide_benzimidazole column from script 08')
        flag = src['retains_sulfoxide_benzimidazole'].astype(str).str.lower().eq('true')
        src = src.loc[flag].copy()
    # Apply the same largest-organic-fragment, atom, size and length
    # policy used in 10_prepare_full_chembl.py before fine-tuning.
    base = load_checkpoint(args.pretrained)
    pretrained_alphabet = set(base['vocab'])
    max_len = min(args.max_len, int(base['config']['max_len']))
    chooser = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
    allowed_atomic_numbers = {1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53}
    smiles_set = set()
    rejected = 0
    fragment_reduced = 0
    incompatible_tokens = 0
    for value in src.canonical_smiles.dropna():
        mol = Chem.MolFromSmiles(str(value))
        if mol is None:
            rejected += 1
            continue
        original_fragments = len(Chem.GetMolFrags(mol))
        mol = chooser.choose(mol)
        if original_fragments > 1:
            fragment_reduced += 1
        heavy = mol.GetNumHeavyAtoms()
        if (heavy < 5 or heavy > 80 or
                any(atom.GetAtomicNum() not in allowed_atomic_numbers
                    for atom in mol.GetAtoms())):
            rejected += 1
            continue
        smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if not smi or len(smi) + 2 > max_len:
            rejected += 1
            continue
        if not set(smi).issubset(pretrained_alphabet):
            incompatible_tokens += 1
            rejected += 1
            continue
        smiles_set.add(smi)
    smiles = sorted(smiles_set)
    if len(smiles) < 20:
        raise ValueError(f'Only {len(smiles)} structures selected; use --selection all or inspect core matches.')
    rng = random.Random(args.seed)
    rng.shuffle(smiles)
    n_val = max(1, int(round(len(smiles) * args.val_fraction)))
    if len(smiles) - n_val < 10:
        raise ValueError('Need at least 10 training molecules')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in [('train', smiles[n_val:]), ('val', smiles[:n_val])]:
        with (out / f'ppi_series_{name}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['canonical_smiles'])
            writer.writeheader()
            writer.writerows([{'canonical_smiles': s} for s in data])
    info = {'selection': args.selection, 'source_rows': len(src), 'unique_valid': len(smiles),
            'train': len(smiles) - n_val, 'val': n_val, 'rejected': rejected,
            'multifragment_rows_standardized': fragment_reduced,
            'incompatible_token_rows_rejected': incompatible_tokens,
            'seed': args.seed, 'source': args.series, 'heldout_note':
            'Random molecule split for fine-tuning diagnostics only, not independent scaffold validation.'}
    (out / 'series_manifest.json').write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


def data_loader(path, stoi, max_len, batch_size, shuffle, epoch, seed):
    ds = pre.StreamSMILES(path, stoi, max_len, shuffle=shuffle, epoch=epoch, seed=seed,
                          buffer=2048)
    return DataLoader(ds, batch_size=batch_size, collate_fn=pre.collate, num_workers=0,
                      pin_memory=torch.cuda.is_available())


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    base = load_checkpoint(args.pretrained)
    vocab, config = base['vocab'], base['config']
    stoi = {s: i for i, s in enumerate(vocab)}
    for fp in [Path(args.out) / 'ppi_series_train.csv', Path(args.out) / 'ppi_series_val.csv']:
        if not fp.exists():
            raise FileNotFoundError(f'{fp}: first run --mode prepare')
        with fp.open() as handle:
            for row in csv.DictReader(handle):
                unknown = set(row['canonical_smiles']) - set(vocab)
                if unknown:
                    raise ValueError(f'Unknown SMILES tokens {unknown} in {fp}. Expand vocabulary with care.')
    model = pre.make_model(config, len(vocab), device)
    model.load_state_dict(base['model_state_dict'], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    out = Path(args.out)
    best = float('inf')
    wait = 0
    history = []
    if args.resume:
        last = load_checkpoint(args.resume)
        if last['vocab'] != vocab or last['config'] != config:
            raise ValueError('Resume checkpoint vocabulary/model differs from pretrained model')
        model.load_state_dict(last['model_state_dict'], strict=True)
        optimizer.load_state_dict(last['optimizer_state_dict'])
        scaler.load_state_dict(last['scaler_state_dict'])
        best = last['best_val_loss']
        wait = last['wait']
        history = last['history']
        start_epoch = last['epoch'] + 1
    else:
        start_epoch = 1
    print(f'Fine-tuning on {device}; parameters={sum(p.numel() for p in model.parameters()):,}; start epoch {start_epoch}')
    for epoch in range(start_epoch, args.epochs + 1):
        tr = data_loader(out / 'ppi_series_train.csv', stoi, config['max_len'], args.batch_size, True, epoch, args.seed)
        va = data_loader(out / 'ppi_series_val.csv', stoi, config['max_len'], args.batch_size, False, 0, args.seed)
        tr_loss, tr_acc, _ = pre.run_epoch(model, tr, device, optimizer=optimizer, scaler=scaler,
                                          amp=device.type == 'cuda', log_every=0)
        va_loss, va_acc, _ = pre.run_epoch(model, va, device, amp=device.type == 'cuda', log_every=0)
        improved = va_loss < best - 1e-4
        if improved:
            best, wait = va_loss, 0
        else:
            wait += 1
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc))
        state = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                     scaler_state_dict=scaler.state_dict(), epoch=epoch, best_val_loss=best,
                     wait=wait, history=history, vocab=vocab, config=config,
                     pretrained_checkpoint=args.pretrained, selection=args.selection)
        tmp = out / 'last.pt.tmp'
        torch.save(state, tmp)
        tmp.replace(out / 'last.pt')
        if improved:
            tmp = out / 'best.pt.tmp'
            torch.save(state, tmp)
            tmp.replace(out / 'best.pt')
        pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
        print(f'Epoch {epoch:02d} | train={tr_loss:.4f} acc={tr_acc:.3f} '
              f'| val={va_loss:.4f} acc={va_acc:.3f}' + (' [BEST]' if improved else ''), flush=True)
        if wait >= args.patience:
            print(f'Early stopping; best val loss={best:.5f}')
            break


def sample(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = load_checkpoint(args.checkpoint or (Path(args.out) / 'best.pt'))
    vocab = ckpt['vocab']
    stoi = {c: i for i, c in enumerate(vocab)}
    model = pre.make_model(ckpt['config'], len(vocab), device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    seen = set()
    rows = []
    for k in range(args.num_samples):
        tokens = [stoi[pre.BOS]]
        with torch.no_grad():
            for j in range(model.max_len - 1):
                x = torch.tensor([tokens], device=device)
                logits = model(x)[0, -1].float() / max(args.temperature, 1e-6)
                for banned in [pre.PAD, pre.BOS, pre.UNK]:
                    logits[stoi[banned]] = -float('inf')
                if args.top_k:
                    cutoff = torch.topk(logits, min(args.top_k, len(vocab))).values[-1]
                    logits[logits < cutoff] = -float('inf')
                next_id = int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())
                if next_id == stoi[pre.EOS]:
                    break
                tokens.append(next_id)
        raw = ''.join(vocab[i] for i in tokens[1:])
        mol = Chem.MolFromSmiles(raw) if raw else None
        valid = mol is not None
        canon = Chem.MolToSmiles(mol, isomericSmiles=True) if valid else ''
        isnew = valid and canon not in seen
        if valid:
            seen.add(canon)
        rows.append({'sample_id': k+1, 'raw_smiles': raw, 'canonical_smiles': canon,
                     'valid': valid, 'unique_in_batch': isnew,
                     'omeprazole_sulfoxide_benzimidazole_motif': bool(valid and motif(mol))})
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f'Saved {path}; valid={sum(r["valid"] for r in rows)}/{len(rows)}; '
          f'core hits={sum(r["omeprazole_sulfoxide_benzimidazole_motif"] for r in rows)}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['prepare', 'train', 'sample'], default='prepare')
    p.add_argument('--series', default='analysis/omeprazole_series/full_chembl37_corrected_matches.csv')
    p.add_argument('--selection', choices=['core', 'all'], default='all')
    p.add_argument('--out', default='checkpoints/ppi_transformer_all')
    p.add_argument('--max-len', type=int, default=202)
    p.add_argument('--val-fraction', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--pretrained', default='checkpoints/smiles_transformer/best.pt')
    p.add_argument('--resume', default='')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--checkpoint', default='')
    p.add_argument('--num-samples', type=int, default=1000)
    p.add_argument('--temperature', type=float, default=0.9)
    p.add_argument('--top-k', type=int, default=30)
    p.add_argument('--output', default='generation/transformer_ppi_finetuned_1000.csv')
    args = p.parse_args()
    {'prepare': prepare, 'train': train, 'sample': sample}[args.mode](args)


if __name__ == '__main__':
    main()
