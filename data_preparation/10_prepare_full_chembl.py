"""Create common full-ChEMBL SMILES corpus for GNN/Transformer/diffusion experiments.

Usage:
 python 10_prepare_full_chembl.py --input data/raw/chembl_37_chemreps.txt.gz --ppi PPI_molecule_summary_248.csv --out data/full_chembl37

Keep molecule stereochemistry and formal charge. Single-fragment organic chemistry only.
Use a stable hash of canonical isomeric SMILES for deterministic split allocation.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from collections import Counter
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

ALLOWED = {1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53}


def standardize_smiles(text, chooser):
    if not isinstance(text, str) or not text.strip():
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    # Largest fragment handles mixtures/salts; preserve atom charge and chirality.
    mol = chooser.choose(mol)
    if mol.GetNumHeavyAtoms() < 5 or mol.GetNumHeavyAtoms() > 80:
        return None
    if any(atom.GetAtomicNum() not in ALLOWED for atom in mol.GetAtoms()):
        return None
    s = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if len(s) > 200 or not s:
        return None
    return s


def stable_bucket(smiles):
    h = hashlib.sha256(smiles.encode('utf-8')).digest()
    return int.from_bytes(h[:8], 'big') % 10000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--ppi', required=True)
    ap.add_argument('--out', default='data/full_chembl37')
    ap.add_argument('--max-rows', type=int, default=None, help='Optional smoke-test cap on source records')
    args = ap.parse_args()
    target = Path(args.out)
    target.mkdir(parents=True, exist_ok=True)
    chooser = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
    ppi_excluded = set()
    with open(args.ppi, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            s = standardize_smiles(row.get('canonical_smiles'), chooser)
            if s:
                ppi_excluded.add(s)
    seen = set()
    counters = Counter()
    paths = {k: target / f'chembl37_{k}.csv' for k in ('train', 'val', 'test')}
    handles = {k: open(path, 'w', newline='', encoding='utf-8') for k, path in paths.items()}
    writers = {k: csv.writer(h) for k, h in handles.items()}
    for w in writers.values():
        w.writerow(['chembl_id', 'canonical_smiles'])
    try:
        opener = gzip.open if str(args.input).endswith('.gz') else open
        with opener(args.input, 'rt', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            assert {'chembl_id', 'canonical_smiles'} <= set(reader.fieldnames or []), reader.fieldnames
            for i, row in enumerate(reader):
                if args.max_rows is not None and i >= args.max_rows:
                    break
                counters['source_rows'] += 1
                smi = standardize_smiles(row['canonical_smiles'], chooser)
                if smi is None:
                    counters['filtered'] += 1
                    continue
                if smi in ppi_excluded:
                    counters['excluded_ppi'] += 1
                    continue
                if smi in seen:
                    counters['duplicates'] += 1
                    continue
                seen.add(smi)
                bucket = stable_bucket(smi)
                split = 'train' if bucket < 9000 else ('val' if bucket < 9500 else 'test')
                writers[split].writerow([row['chembl_id'], smi])
                counters[split] += 1
    finally:
        for h in handles.values():
            h.close()
    manifest = {
        'source': str(args.input), 'ppi_file': str(args.ppi),
        'rdkit_version': rdBase.rdkitVersion,
        'policy': 'largest organic fragment; preserve formal charge and stereochemistry; allowed atoms H,B,C,N,O,F,P,S,Cl,Br,I; heavy atoms 5-80; SMILES length <=200; dedupe by canonical isomeric SMILES; exclude exact standardized PPI structures',
        'split': 'SHA256 SMILES 90/5/5 deterministic train/val/test',
        'counts': dict(counters), 'outputs': {k: str(v) for k, v in paths.items()}
    }
    with open(target / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
