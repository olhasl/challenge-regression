import pandas as pd
import os

# Read the CSV files into a DataFrame
code_dir = os.path.dirname(os.path.realpath(__file__))  # Directory of the script
ready_initial_data_path = os.path.join(code_dir, "../input_data/ready_initial_data.csv")
df1 = pd.read_csv(ready_initial_data_path)

code_nis_path = os.path.join(code_dir, "../input_data/georef-belgium-postal-codes.csv")
df2 = pd.read_csv(code_nis_path, header=0)

df1["Zip Code"] = df1["Zip Code"].astype(str)
df2["Post code"] = df2["Post code"].astype(str)
df2["Municipality code"] = df2["Municipality code"].astype(int)
df2["Municipality code"] = df2["Municipality code"].astype(str)
df2.drop_duplicates(subset=["Post code"], inplace=True)

# Adding nisCode column
# Merging files by a common column with zip code
merged_df = pd.merge(
    df1,
    df2[["Post code", "Municipality code"]],
    how="left",
    left_on="Zip Code",
    right_on="Post code",
)
merged_df.drop(columns=["Post code"], inplace=True)
merged_df = merged_df.rename(columns={"Municipality code": "nisCode"})
merged_df["nisCode"] = merged_df["nisCode"].astype(str)

# Adding Population column
population_path = os.path.join(code_dir, "../input_data/TF_SOC_POP_STRUCT_2024.csv")
df3 = pd.read_csv(population_path, header=0)
df3["CD_REFNIS"] = df3["CD_REFNIS"].astype(str)
df3["MS_POPULATION"] = df3["MS_POPULATION"].astype(int)
df3_grouped = df3.groupby("CD_REFNIS")["MS_POPULATION"].sum().reset_index()
# Merging files by a common column with nis code
merged_df = merged_df.merge(
    df3_grouped[["CD_REFNIS", "MS_POPULATION"]],
    how="left",
    left_on="nisCode",
    right_on="CD_REFNIS",
)
# Removing an unnecessary column
merged_df.drop(columns=["CD_REFNIS"], inplace=True)
merged_df = merged_df.rename(columns={"MS_POPULATION": "Population"})

# Adding Taxable Income column
income_path = os.path.join(code_dir, "../input_data/TF_PSNL_INC_TAX_MUNTY.csv")
df4 = pd.read_csv(income_path, header=0)
df4 = df4.loc[df4["CD_YEAR"] == 2022]
df4["CD_MUNTY_REFNIS"] = df4["CD_MUNTY_REFNIS"].astype(str)
df4["MS_TOT_NET_TAXABLE_INC"] = df4["MS_TOT_NET_TAXABLE_INC"].astype(float)
df4["MS_NBR_NON_ZERO_INC"] = df4["MS_NBR_NON_ZERO_INC"].astype(int)
df4["Avg_Taxable_Income"] = round(
    df4["MS_TOT_NET_TAXABLE_INC"] / df4["MS_NBR_NON_ZERO_INC"], 2
)
# Merging files by a common column with nis code
merged_df = merged_df.merge(
    df4[["CD_MUNTY_REFNIS", "Avg_Taxable_Income"]],
    how="left",
    left_on="nisCode",
    right_on="CD_MUNTY_REFNIS",
)
# Removing an unnecessary column
merged_df.drop(columns=["CD_MUNTY_REFNIS"], inplace=True)

# Adding Number_of_Buildings column
buildings_path = os.path.join(
    code_dir, "../input_data/building_stock_open_data_2024.csv"
)
df5 = pd.read_csv(buildings_path, header=0)
df5["CD_REFNIS"] = df5["CD_REFNIS"].astype(str)
df5_grouped = df5.groupby("CD_REFNIS")["MS_VALUE"].sum().reset_index()
# Merging files by a common column with nis code
merged_df = merged_df.merge(
    df5_grouped[["CD_REFNIS", "MS_VALUE"]],
    how="left",
    left_on="nisCode",
    right_on="CD_REFNIS",
)
# Removing an unnecessary column
merged_df.drop(columns=["CD_REFNIS"], inplace=True)
merged_df = merged_df.rename(columns={"MS_VALUE": "Number_of_Buildings"})

# Adding Number_of_Transactions column
transactions_path = os.path.join(code_dir, "../input_data/vastgoed_2010_9999.xlsx")
all_sheets = pd.read_excel(transactions_path, sheet_name=None)
df6 = pd.concat(all_sheets.values(), ignore_index=True)
df6["CD_REFNIS"] = df5["CD_REFNIS"].astype(str)
df6_grouped = df6.groupby("CD_REFNIS")["MS_TOTAL_TRANSACTIONS"].sum().reset_index()
# Merging files by a common column with nis code
merged_df = merged_df.merge(
    df6_grouped[["CD_REFNIS", "MS_TOTAL_TRANSACTIONS"]],
    how="left",
    left_on="nisCode",
    right_on="CD_REFNIS",
)
# Removing an unnecessary column
merged_df.drop(columns=["CD_REFNIS"], inplace=True)
merged_df = merged_df.rename(
    columns={"MS_TOTAL_TRANSACTIONS": "Number_of_Transactions"}
)
merged_df["Number_of_Transactions"] = merged_df["Number_of_Transactions"].fillna(0)
merged_df["Number_of_Transactions"] = merged_df["Number_of_Transactions"].astype(int)

# Specify the feature columns
add_features_columns = [
    "Fully Equipped Kitchen",
    "Furnished",
    "Open Fire",
    "Terrace",
    "Garden",
    "Swimming Pool",
]
# Create "Add_Fearures" column indicating whether any of the additional features are present
merged_df["Add_Fearures"] = merged_df[add_features_columns].any(axis=1)
merged_df["Add_Fearures"] = merged_df["Add_Fearures"].apply(
    lambda x: "yes" if x else "no"
)
# Convert "Add_Fearures" into dummy variables
merged_df = pd.get_dummies(
    merged_df, columns=["Add_Fearures"], prefix="Add_Fearures", drop_first=True
)
merged_df = merged_df.rename(columns={"Add_Fearures_yes": "Add_Fearures"})

# Drop unnecessary columns
columns_to_drop = [
    "Locality",
    "Zip Code",
    "nisCode",
    "Fully Equipped Kitchen",
    "Furnished",
    "Open Fire",
    "Terrace",
    "Garden",
    "Swimming Pool",
    "Terrace Area (m2)",
    "Garden Area (m2)",
    "Surface of the Land (m2)",
    "Number of Facades",
]
merged_df = merged_df.drop(
    columns=[col for col in columns_to_drop if col in merged_df.columns]
)

# Save the new DataFrame to CSV file
data_for_model_path = os.path.join(code_dir, "../output_data/data_for_model.csv")
merged_df.to_csv(data_for_model_path, index=False)
