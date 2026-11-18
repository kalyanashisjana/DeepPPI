
from pathlib import Path
import pandas as pd

# Input file
input_file = Path(
    "generation/transformer_ppi_finetuned_evaluated.csv"
)

# Output directory
output_dir = Path("generation/unique_core_molecules")
output_dir.mkdir(parents=True, exist_ok=True)

# Load generated molecules
df = pd.read_csv(input_file)

# Select valid molecules retaining the PPI core
core_df = df[
    (df["valid_checked"] == True) &
    (df["omeprazole_sulfoxide_benzimidazole_motif"] == True)
].copy()

print("Total core-retaining samples:", len(core_df))

# Remove duplicates using canonical SMILES
unique_core = core_df.drop_duplicates(
    subset="canonical_smiles_checked"
).copy()

unique_core = unique_core.reset_index(drop=True)

# Assign identifiers
unique_core.insert(
    0,
    "molecule_id",
    [f"PPI_GEN_{i:04d}" for i in range(1, len(unique_core) + 1)]
)

# Save complete CSV
csv_path = output_dir / "PPI_116_unique_core.csv"
unique_core.to_csv(csv_path, index=False)

# Save SMILES with molecule identifiers
smi_path = output_dir / "PPI_116_unique_core.smi"

with open(smi_path, "w") as f:
    for row in unique_core.itertuples():
        f.write(
            f"{row.canonical_smiles_checked}\t{row.molecule_id}\n"
        )

# Save only SMILES as a text file
txt_path = output_dir / "PPI_116_unique_core.txt"

unique_core["canonical_smiles_checked"].to_csv(
    txt_path,
    index=False,
    header=False
)

print("\nExtraction completed!")
print("Unique core-retaining molecules:", len(unique_core))
print("CSV saved:", csv_path)
print("SMILES saved:", smi_path)
print("TXT saved:", txt_path)
