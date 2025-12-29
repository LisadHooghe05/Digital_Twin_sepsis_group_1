import pandas as pd
from pathlib import Path

# Path to data
REPO_ROOT = Path(__file__).resolve().parent.parent
PATH_DATA = REPO_ROOT / "data"

# Load AKI data
aki_output = pd.read_csv("AKI_stage_output.csv")
aki_output["subject_id"] = aki_output["subject_id"].astype(str)
aki_output["AKI_time"] = pd.to_datetime(aki_output["AKI_time"], errors='coerce')

# Load creatinine values
creatinine_data = pd.read_csv(PATH_DATA / "creatinine.csv")
creatinine_data["subject_id"] = creatinine_data["subject_id"].astype(str)
creatinine_data["charttime"] = pd.to_datetime(creatinine_data["charttime"], errors='coerce')

# Merge the creatinine data with the AKI output 
merged_data = pd.merge(creatinine_data, aki_output[['subject_id', 'AKI_time','baseline_value']], on='subject_id', how='left')

# Compute the time difference between creatinine and AKI time
merged_data['time_diff'] = (merged_data['charttime'] - merged_data['AKI_time']).dt.total_seconds()

# Filter creatinine data 12 hours before AKI time and 12 hours after AKI time
filtered_data = merged_data[
    (merged_data['time_diff'] >= -12*3600) &  # 12 hours before AKI time
    (merged_data['time_diff'] <= 12*3600)    # 12 hours after AKI time
]

# Select only wanted columns
filtered_data = filtered_data[['subject_id', 'charttime', 'creatinine_mg_dl','baseline_value', 'AKI_time']]


# Selete rows where 'AKI_time' NaN is (no AKI Time)
filtered_data = filtered_data.dropna(subset=['AKI_time'])

REPO_ROOT = Path(__file__).resolve().parent.parent
PATH_CSV = REPO_ROOT / "csv_dashboard"

# Look if the map reallye exists
PATH_CSV.mkdir(exist_ok=True)

# Save to csv
output_path = PATH_CSV / "creatinine_filtered.csv"
filtered_data.to_csv(output_path, index=False,decimal=',')

# Reset index for nicer dataframe
filtered_data.reset_index(drop=True, inplace=True)

# Toon de gefilterde data
print(filtered_data.head())
