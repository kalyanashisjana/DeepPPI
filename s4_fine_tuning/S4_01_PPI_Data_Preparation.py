#!/usr/bin/env python3
"""Create S4-compatible PPI SMILES files from the corrected ChEMBL analogue CSV.

Requires: pandas, rdkit, s4dd. Uses the actual S4 checkpoint token vocabulary.
The S4 model uses tokenized SMILES and cannot automatically learn new tokens
while fine-tuning an already pretrained embedding matrix.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from s4dd import S4forDenovoDesign
from s4dd.smiles_utils import segment_smiles

RDLogger.DisableLog('rdApp.error')


def prep_smiles(s, normalize_sulfoxide=True, strip_stereo=True):
    if not isinstance(s, str) or not s.strip():
        return None
    mol = Chem.MolFromSmiles(s.strip())
    if mol is None:
        return None
    # Single largest organic fragment, consistently with the earlier PPI Transformer preparation.
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    organic = [f for f in frags if any(a.GetAtomicNum() == 6 for a in f.GetAtoms())]
    if not organic:
        return None
    mol = max(organic, key=lambda f: (f.GetNumHeavyAtoms(), f.GetNumAtoms()))
    # The ChEMBL-v31 S4 model may not contain charge tokens needed for
    # [S+]([O-]) sulfoxides. Convert *this precise resonance depiction* to S(=O).
    # This is not general neutralization of charged molecules.
    if normalize_sulfoxide:
        rw = Chem.RWMol(mol)
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() != 16 or atom.GetFormalCharge() != 1:
                continue
            for neighbor in atom.GetNeighbors():
                if (neighbor.GetAtomicNum() == 8 and neighbor.GetFormalCharge() == -1
                        and rw.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType() == Chem.BondType.SINGLE):
                    rw.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).SetBondType(Chem.BondType.DOUBLE)
                    atom.SetFormalCharge(0)
                    neighbor.SetFormalCharge(0)
                    break
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
    if strip_stereo:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=not strip_stereo, canonical=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Corrected analogue CSV (802 source records)')
    ap.add_argument('--checkpoint', required=True, help='Official ChEMBL S4 model directory')
    ap.add_argument('--out', default='experiments/s4_ppi786/data')
    ap.add_argument('--column', default='canonical_smiles')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--preserve-stereo', action='store_true', help='Keep stereochemical tokens (may not exist in pretrained S4 vocabulary)')
    ap.add_argument('--no-normalize-sulfoxide', action='store_true', help='Do not convert [S+]([O-]) representation to S(=O)')
    ap.add_argument('--val-frac', type=float, default=.10)
    args = ap.parse_args()
    if not 0 < args.val_frac < .5:
        ap.error('--val-frac must be between 0 and 0.5')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.checkpoint)
    if not (model_dir/'model.pt').exists() or not (model_dir/'init_arguments.json').exists():
        ap.error('Checkpoint directory must contain model.pt and init_arguments.json')
    # Obtain vocabulary and max token length from the saved model metadata, without loading GPU weights.
    props = json.loads((model_dir/'init_arguments.json').read_text())
    vocab = set(props['token2label'])
    sequence_length = int(props['sequence_length'])
    source = pd.read_csv(args.input)
    if args.column not in source:
        ap.error(f'Column {args.column!r} not found: {list(source.columns)}')
    counts = Counter()
    accepted = []
    seen = set()
    for raw in source[args.column].tolist():
        counts['source_rows'] += 1
        smiles = prep_smiles(raw, normalize_sulfoxide=not args.no_normalize_sulfoxide, strip_stereo=not args.preserve_stereo)
        if smiles is None:
            counts['invalid_or_nonorganic'] += 1
            continue
        if smiles in seen:
            counts['duplicate_after_standardization'] += 1
            continue
        tokens = segment_smiles(smiles)
        # Verify the tokenizer has not silently dropped a character.
        if ''.join(tokens) != smiles:
            counts['tokenization_not_reversible'] += 1
            continue
        if any(t not in vocab for t in tokens):
            counts['unsupported_S4_token'] += 1
            continue
        # S4 creates [BEG] + tokens + [END] and pads/truncates to sequence_length + 1.
        if len(tokens) > sequence_length - 1:
            counts['sequence_too_long'] += 1
            continue
        seen.add(smiles)
        accepted.append(smiles)
    rng = random.Random(args.seed)
    rng.shuffle(accepted)
    n_val = max(1, round(args.val_frac * len(accepted))) if len(accepted) >= 2 else 0
    if not n_val or len(accepted) < 10:
        raise SystemExit(f'Only {len(accepted)} compatible unique molecules found; inspect rejection counts: {dict(counts)}')
    val = accepted[:n_val]
    train = accepted[n_val:]
    (out/'train.txt').write_text('\n'.join(train) + '\n')
    (out/'valid.txt').write_text('\n'.join(val) + '\n')
    pd.DataFrame({'canonical_smiles': train, 'partition': 'train'}).to_csv(out/'train.csv', index=False)
    pd.DataFrame({'canonical_smiles': val, 'partition': 'val'}).to_csv(out/'valid.csv', index=False)
    summary = dict(source=str(Path(args.input).resolve()), checkpoint=str(model_dir.resolve()),
                   seed=args.seed, train=len(train), val=len(val), usable=len(accepted),
                   rejections=dict(counts), sequence_length=sequence_length,
                   representation='S(=O) sulfoxide resonance form by default; stereochemistry removed by default',
                   note='S4 model training uses structures, not activity labels. Other charged groups are retained and can be rejected if vocabulary does not support them.')
    (out/'manifest.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'Saved {out / "train.txt"} and {out / "valid.txt"}')

if __name__ == '__main__':
    main()
