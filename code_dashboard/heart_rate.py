import pandas as pd
from pathlib import Path

def heartrate():
    """
    Returns a long-format dataframe:
    subject_id | hours_before_AKI | label | valuenum
    filtered to Heart Rate only
    sorted chronologically (12h → 0h)
    """

    # Paths
    PATH_DATA = Path(__file__).resolve().parent.parent / "data"
    vitals_files = [PATH_DATA / "vitals1.csv", PATH_DATA / "vitals2.csv", PATH_DATA / "vitals3.csv"]
    sepsis_file = PATH_DATA / "sepsis_diagnose_time.csv"
    aki_file = "AKI_subjects.csv"
    aki_times_file = "AKI_stage_output.csv"

    # Read sepsis mapping + AKI subjects
    sepsis_df = pd.read_csv(sepsis_file, dtype={'stay_id': str, 'subject_id': str})
    aki_stage_df = pd.read_csv(aki_file, dtype={'subject_id': str})
    aki_subjects = set(aki_stage_df['subject_id'])

    aki_times_df = pd.read_csv(aki_times_file, dtype={'subject_id': str})
    aki_times_df['AKI_time'] = pd.to_datetime(aki_times_df['AKI_time'], errors='coerce')

    # Read and merge vitals
    vitals_list = [pd.read_csv(vf, dtype={'stay_id': str}) for vf in vitals_files]
    vitals_df = pd.concat(vitals_list, ignore_index=True)
    vitals_df = vitals_df.merge(sepsis_df[['stay_id', 'subject_id']], on='stay_id', how='left')
    vitals_df = vitals_df[vitals_df['subject_id'].isin(aki_subjects)]

    vitals_df['charttime'] = pd.to_datetime(vitals_df['charttime'], errors='coerce')
    vitals_df = vitals_df.merge(aki_times_df[['subject_id', 'AKI_time']], on='subject_id', how='left')

    # Compute hours before AKI
    vitals_df['hours_before_AKI'] = (vitals_df['AKI_time'] - vitals_df['charttime']).dt.total_seconds() / 3600
    df12 = vitals_df[(vitals_df['hours_before_AKI'] >= 0) & (vitals_df['hours_before_AKI'] <= 12)]

    # Keep only Heart Rate
    df12 = df12[df12['label'] == 'Heart Rate']

    # Filter out extreme values
    df12 = df12[df12['valuenum'].between(20, 300)]

    # Sort chronologically
    df12 = df12.sort_values(by=['subject_id', 'hours_before_AKI'], ascending=[True, False])
    df12 = df12.round(2)

    return df12


# Generate vitals dataframe
df = heartrate()

# Convert to wide format
out = df.pivot_table(
    index=['subject_id', 'hours_before_AKI'],
    columns='label',
    values='valuenum',
    aggfunc='mean'
).reset_index()

out = out.sort_values(['subject_id', 'hours_before_AKI'], ascending=[True, False])

# Paths for saving
REPO_ROOT = Path(__file__).resolve().parent.parent
PATH_CSV = REPO_ROOT / "csv_dashboard"
PATH_CSV.mkdir(exist_ok=True)

# Save vitals chronologically
output_path = PATH_CSV / "vitals_chronological.csv"
out.to_csv(output_path, index=False, decimal=',')

# Read back using correct path
vitals = pd.read_csv(output_path)

# Save unique subject_ids
subject_lookup = vitals[['subject_id']].drop_duplicates()
subject_lookup.to_csv(PATH_CSV / "subject_ids.csv", index=False)
