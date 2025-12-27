import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
PATH_DATA = REPO_ROOT / "data"
SEPSIS_CSV  = "sepsis_diagnose_time.csv"
CREAT_CSV   = "creatinine_over_time.csv"

from Tool_1.determining_AKI import AKI_detection

Sepsis_AKI_stages = AKI_detection(SEPSIS_CSV,CREAT_CSV)
Sepsis_AKI_stages["AKI_stage"] = Sepsis_AKI_stages["AKI_stage"].replace({
    "AKI stage 1 (48h rise ≥0.3 mg/dL)": "AKI stage 1"})
def get_column_index(df, column_name):
    """
    Returns the position of a wanted column in the datafrome
    if the column does not exist, raises a ValueError
    """
    try: 
        return df.columns.get_loc(column_name)
    except KeyError:
        raise ValueError(f"Column'{column_name}' not found in Dataframe")


df = Sepsis_AKI_stages 
col_name = "AKI_stage" # -> can you change to the column name from which you want the data
col_index = get_column_index(df,col_name)
print(f"Column'{col_name}' is at index '{col_index}'")

def different_stages_factor(df: pd.DataFrame,kolom_index:int):
    """
    Uses a dataframe and takes the column identified by its index
    Returns a dictionary with unique keys and their corresponding rows and values
    """
    specific_column = df.columns[kolom_index]
    column_data = df[specific_column]
    different_kinds = column_data.unique()
    print(f"Unique kinds found in '{specific_column}': {different_kinds}")

    sorted_dict = {}
    for kind in different_kinds:
        sorted_dict[kind] =df[df[specific_column]==kind]
    return sorted_dict

result = different_stages_factor(Sepsis_AKI_stages,col_index)
result = different_stages_factor(Sepsis_AKI_stages,12)
print("DEBUG: result keys:", result.keys())
for key, df in result.items():
    print(f"{key}: {len(df)} rows")

# Making lists
stages = []
counts = []
total_amount = 0

for key, value in result.items():
    stages.append(key)
    counts.append(len(value)) # amount of rows per group
    total_amount += len(value)

print(f"Debug: Total amount of patients divided into different stages: '{total_amount}'") # to check if the amount of stages is equal to the amount of sepsis patients

desired_order_without_filtering = [
    "AKI determination not possible",
    "No AKI",
    "AKI stage 1 (48h rise ≥0.3 mg/dL)",
    "AKI stage 1",
    "AKI stage 2",
    "AKI stage 3"]

desired_order = [
    "AKI determination not possible",
    "No AKI",
    "AKI stage 1",
    "AKI stage 2",
    "AKI stage 3"]

stages_sorted = [stage for stage in desired_order_without_filtering if stage in result.keys()]
counts_sorted = [len(result[stage]) for stage in stages_sorted]

#making a plot
# plt.figure(figsize=(10,7))
# bars = plt.bar(stages_sorted,counts_sorted)
# plt.bar_label(bars)
# plt.xticks(rotation=45,ha="right")
# plt.xlabel("AKI Stage")
# plt.ylabel("Amount of patients")
# plt.title("Distribution of AKI stages")
# plt.tight_layout()
# plt.show(block=True)


# Make CSV with unique subject_id per stage
df_ids = Sepsis_AKI_stages[['subject_id', 'AKI_stage']].drop_duplicates()
df_ids = df_ids.sort_values(['AKI_stage', 'subject_id'])
df_ids.to_csv("AKI_stages_subject.csv", index=False)

stages_sorted_filtered = [stage for stage in desired_order if stage in df_ids["AKI_stage"].unique()]
counts_sorted_filtered = [df_ids[df_ids["AKI_stage"] == stage].shape[0] for stage in stages_sorted_filtered]

# Print the distribution and total number of unique subject_ids
print(df_ids["AKI_stage"].value_counts())
print("Aantal unieke subject_ID's:", df_ids["subject_id"].nunique())


plt.figure(figsize=(10,7))
bars2 = plt.bar(stages_sorted_filtered,counts_sorted_filtered)
plt.bar_label(bars2)
plt.xticks(rotation=45,ha="right")
plt.xlabel("AKI Stage")
plt.ylabel("Amount of patients")
plt.title("Distribution of AKI stages filtered")
plt.tight_layout()
plt.show(block=True)
