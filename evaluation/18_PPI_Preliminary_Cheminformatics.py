#!/usr/bin/env python3
"""RDKit-only profiling and novelty check for generated PPI molecules.
Input: CSV with canonical_smiles_checked, canonical_smiles, or smiles.
Usage: python 18_PPI_Preliminary_Cheminformatics.py --input generation/unique_core_molecules/PPI_116_unique_core.csv
"""
import argparse, csv
from pathlib import Path
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.error')


def read_smiles_col(path):
    with open(path,newline='',encoding='utf-8') as f:
        rdr=csv.DictReader(f)
        col=next((c for c in ['canonical_smiles_checked','canonical_smiles','smiles','SMILES'] if c in rdr.fieldnames),None)
        if col is None:raise ValueError(f'No SMILES column in {path}; found {rdr.fieldnames}')
        for row in rdr:
            s=row.get(col,'')
            if s and isinstance(s,str): yield row,s


def norm(s):
    m=Chem.MolFromSmiles(s)
    if m is None:return None,None
    return m,Chem.MolToSmiles(m,isomericSmiles=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='generation/unique_core_molecules/PPI_116_unique_core.csv')
    p.add_argument('--ppi',default='PPI_molecule_summary_248.csv')
    p.add_argument('--chembl',default='data/full_chembl37/chembl37_train.csv')
    p.add_argument('--output',default='generation/unique_core_molecules/PPI_116_cheminformatics.csv')
    a=p.parse_args()
    refs=[]
    gen=AllChem.GetMorganGenerator(radius=2,fpSize=2048)
    for _,s in read_smiles_col(a.ppi):
        m,can=norm(s)
        if m: refs.append((can,gen.GetFingerprint(m)))
    gen=AllChem.GetMorganGenerator(radius=2,fpSize=2048)
    result=[]
    for i,(row,s) in enumerate(read_smiles_col(a.input)):
        m,can=norm(s)
        if m is None:continue
        fp=gen.GetFingerprint(m)
        sim=max((DataStructs.TanimotoSimilarity(fp,f) for _,f in refs),default=float('nan'))
        matches=sum(1 for ref,_ in refs if ref==can)
        d=dict(row)
        d.update(canonical_smiles=can,MW=round(Descriptors.MolWt(m),2),LogP=round(Crippen.MolLogP(m),3),
                 TPSA=round(rdMolDescriptors.CalcTPSA(m),2),HBD=rdMolDescriptors.CalcNumHBD(m),
                 HBA=rdMolDescriptors.CalcNumHBA(m),RotBonds=rdMolDescriptors.CalcNumRotatableBonds(m),
                 QED=round(QED.qed(m),3),formal_charge=Chem.GetFormalCharge(m),
                 murcko_scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=m),
                 max_PPI_Tanimoto=round(sim,3),exact_labeled_PPI=matches>0,
                 exact_ChEMBL_train=False)
        result.append(d)
    wanted=set(r['canonical_smiles'] for r in result)
    observed=set()
    print(f'Checking {len(wanted)} generated structures against {a.chembl} ...',flush=True)
    # Exact canonical isomeric SMILES lookup; no extra structure normalization.
    for _,s in read_smiles_col(a.chembl):
        if s in wanted:observed.add(s)
    for r in result:r['exact_ChEMBL_train']=r['canonical_smiles'] in observed
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    if result:
        with open(out,'w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    print(f'Valid structures profiled: {len(result)}')
    print(f'Exact matches ChEMBL training: {sum(r["exact_ChEMBL_train"] for r in result)}')
    print(f'Exact matches labeled PPI set: {sum(r["exact_labeled_PPI"] for r in result)}')
    print(f'Unique Murcko scaffolds: {len(set(r["murcko_scaffold"] for r in result))}')
    print(f'Output: {out}')
    print('This does not establish experimental activity, mechanistic suitability, or novelty vs all databases.')

if __name__=='__main__':main()
