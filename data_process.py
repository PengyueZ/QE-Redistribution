import pandas as pd

# Load SCF 2022 data (ensure the file path is correct)
df = pd.read_stata("scf2022.dta")

# Define short-term and long-term asset variables
short_term_vars = ["x3529", "x3730", "x3736", "x3742", "x3748", "x3754", "x3760", "x3765", "x3930"]
long_term_vars = ["x3721", "x7636", "x7637", "x7635", "x7638", "x7639", "x3915"]

# Create short-term and long-term asset totals
df["short_term"] = df[short_term_vars].fillna(0).sum(axis=1)
df["long_term"] = df[long_term_vars].fillna(0).sum(axis=1)
df["total_assets"] = df["short_term"] + df["long_term"]

# Filter out households with zero financial assets
df_filtered = df[df["total_assets"] > 0].copy()

# Compute portfolio shares
df_filtered["share_short"] = df_filtered["short_term"] / df_filtered["total_assets"]
df_filtered["share_long"] = df_filtered["long_term"] / df_filtered["total_assets"]

# Classify household types
df_filtered["type"] = df_filtered.apply(
    lambda row: "short-preferring" if row["share_short"] > row["share_long"] else "long-preferring", axis=1
)

# Compute "short-preferring" total assets share
short_preferring_asset_share = (
    df_filtered.loc[df_filtered["type"] == "short-preferring", "total_assets"] * 
    df_filtered.loc[df_filtered["type"] == "short-preferring", "x42001"]
).sum() / (
    df_filtered["total_assets"] * df_filtered["x42001"]
).sum()

print(f"Total Asset Share of Short-Preferring Households: {short_preferring_asset_share:.2%}")

# Normalize weights
df_filtered["x42001"] = df_filtered["x42001"] / 1e6

# Compute weighted summary
summary = df_filtered.groupby("type").apply(
    lambda g: pd.Series({
        "Avg Share Short": (g["share_short"] * g["x42001"]).sum() / g["x42001"].sum(),
        "Avg Share Long": (g["share_long"] * g["x42001"]).sum() / g["x42001"].sum(),
        "Population Share": g["x42001"].sum() / df_filtered["x42001"].sum() * 100
    })
).reset_index()

print(summary)

# Compute total income for each type
income_summary = df_filtered.groupby("type").apply(
    lambda g: pd.Series({
        "Weighted Average Income": (g["x5729"] * g["x42001"]).sum() / g["x42001"].sum(),
        "Total Income": (g["x5729"] * g["x42001"]).sum(),
        "Population Share": g["x42001"].sum() / df_filtered["x42001"].sum() * 100
    })
).reset_index()

# Compute ratio
income_ratio = (
    income_summary.loc[income_summary["type"] == "long-preferring", "Weighted Average Income"].values[0] /
    income_summary.loc[income_summary["type"] == "short-preferring", "Weighted Average Income"].values[0]
)
print(income_summary)
print(f"Income Ratio (Long/Short): {income_ratio:.2f}")
