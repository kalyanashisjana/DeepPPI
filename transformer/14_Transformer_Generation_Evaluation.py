#!/usr/bin/env python3
"""Evaluate SMILES Transformer samples against training ChEMBL and known PPI molecules.

Run from the GNN project directory; see --help for all arguments.
This script reports *structural* metrics, never experimental potency.
"""
import argparse
from pathlib import Path
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Crippen, QED, rdMolDescriptors
RDLogger.DisableLog('rdApp.warning')

CORE_SMARTS = (
    '[#6]-[S;X3](=[O])-[c]1nc2ccccc2[nH]1',
    '[#6]-[S+;X3]([O-])-[c]1nc2ccccc2[nH]1'
)
CORES = [Chem.MolFromSmarts(s) for s in CORE_SMARTS]

def canonical(s):
    if not isinstance(s, str) or not s.strip():
        return None
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, isomericSmiles=True) if m else None

def get_smiles_column(df):
    for c in ('canonical_smiles', 'smiles', 'SMILES', 'canonical_smiles_standardized', 'molecule_smiles'):
        if c in df.columns: return c
    raise ValueError(f'No SMILES column in {list(df.columns)}')

def fp(m):
    return AllChem.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(m)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', default='generation/transformer_unconditional.csv')
    ap.add_argument('--train', default='data/full_chembl37/chembl37_train.csv')
    ap.add_argument('--ppi', default='PPI_molecule_summary_248.csv')
    ap.add_argument('--out', default='generation/transformer_evaluation.csv')
    ap.add_argument('--no-chEMBL-lookup', action='store_true', help='Skip exact overlap with the 2.46M training molecules')
    args = ap.parse_args()
    samples = pd.read_csv(args.samples)
    sc = get_smiles_column(samples)
    print(f'Samples: {len(samples):,}')
    entries = []
    for _, r in samples.iterrows():
        c = canonical(r[sc])
        m = Chem.MolFromSmiles(c) if c else None
        entries.append((c,m))
    valid_canon = {c for c,m in entries if m is not None}
    train_hits = set()
    if not args.no_chEMBL_lookup:
        # Memory efficient: only retain exact intersections with the generated molecules.
        print('Streaming ChEMBL training CSV to check exact overlaps...')
        for chunk in pd.read_csv(args.train, usecols=['canonical_smiles'], chunksize=100_000):
            train_hits.update(valid_canon.intersection(chunk['canonical_smiles'].dropna().astype(str)))
    pp = pd.read_csv(args.ppi)
    pc = get_smiles_column(pp)
    ppi_mols = []
    for s in pp[pc].dropna():
        c=canonical(s)
        if c:
            ppi_mols.append(Chem.MolFromSmiles(c))
    ppi_fps = [fp(m) for m in ppi_mols]
    ppi_set = {Chem.MolToSmiles(m, isomericSmiles=True) for m in ppi_mols}
    print('Reference PPI structures:',len(ppi_mols))
    rows = []
    for i, (c,m) in enumerate(entries):
        base = samples.iloc[i].to_dict()
        base['canonical_smiles_checked']=c
        base['valid_checked']=m is not None
        base['in_training_corpus'] = (c in train_hits) if m is not None and not args.no_chEMBL_lookup else None
        base['known_ppi_exact'] = c in ppi_set if m is not None else None
        if m is not None:
            q = fp(m)
            base.update({
                'max_similarity_to_248_ppis': max(DataStructs.BulkTanimotoSimilarity(q,ppi_fps)) if ppi_fps else None,
                'omeprazole_sulfoxide_benzimidazole_motif': any(m.HasSubstructMatch(core) for core in CORES),
                'MW': Descriptors.MolWt(m),
                'LogP': Crippen.MolLogP(m),
                'TPSA': rdMolDescriptors.CalcTPSA(m),
                'QED': QED.qed(m),
            })
        rows.append(base)
    res = pd.DataFrame(rows)
    output=Path(args.out);output.parent.mkdir(parents=True,exist_ok=True)
    res.to_csv(output,index=False)
    v=res[res['valid_checked']]
    print(f'Valid: {len(v):,}/{len(res):,} ({len(v)/max(1,len(res)):.1%})')
    print(f'Unique valid: {v.canonical_smiles_checked.nunique():,}/{len(v):,}')
    if not args.no_chEMBL_lookup:
        print('Exact matches to ChEMBL train:',int(v.in_training_corpus.sum()))
        print('Novel vs ChEMBL train (exact canonical):',int((~v.in_training_corpus.astype(bool)).sum()))
    print('Matches to known PPI structures:',int(v.known_ppi_exact.sum()))
    print('Omeprazole sulfoxide-benzimidazole motif hits:',int(v.omeprazole_sulfoxide_benzimidazole_motif.sum()))
    print('PPI max Morgan similarity median:',round(v.max_similarity_to_248_ppis.median(),3))
    print('PPI similarity >= 0.5:',int((v.max_similarity_to_248_ppis >= 0.5).sum()))
    print('Saved:',output)

if __name__=='__main__': main()
