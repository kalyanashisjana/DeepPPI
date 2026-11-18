from chembl_webresource_client.new_client import new_client
import pandas as pd

target_id = "CHEMBL2095173"

# Retrieve bioactivity measurements
activities = new_client.activity.filter(
    target_chembl_id=target_id
).only(
    "molecule_chembl_id",
    "canonical_smiles",
    "standard_type",
    "standard_value",
    "standard_units",
    "standard_relation",
    "assay_chembl_id",
    "target_chembl_id"
)

df = pd.DataFrame(list(activities))

print("Total activity records:", len(df))
print("Unique molecules:", df["molecule_chembl_id"].nunique())

print("\nMeasurement types:")
print(df["standard_type"].value_counts())

# Select quantitative IC50 measurements
df_ic50 = df[
    (df["standard_type"] == "IC50")
    & (df["standard_units"] == "nM")
    & (df["standard_relation"] == "=")
].copy()

df_ic50["standard_value"] = pd.to_numeric(
    df_ic50["standard_value"], errors="coerce"
)

df_ic50 = df_ic50[
    (df_ic50["standard_value"] > 0)
    & df_ic50["canonical_smiles"].notna()
]

print("\nUsable IC50 records:", len(df_ic50))
print("Unique IC50 molecules:",
      df_ic50["molecule_chembl_id"].nunique())

df_ic50.to_csv("PPI_IC50_raw.csv", index=False)
