import pandas as pd
from pathlib import Path

def get_age_12h_before_AKI():
    """
    Returns a DataFrame with one age per subject, closest to 12h before AKI onset.
    Uses internal file paths.
    """

    # Paths
    REPO_ROOT = Path(__file__).resolve().parent.parent
    PATH_DATA = REPO_ROOT / "data"

    age_csv_path = PATH_DATA / "age.csv"  # CSV with columns: ['subject_id', 'time', 'age']
    aki_csv_path = "AKI_stage_output.csv"  # CSV with columns: ['subject_id', 'AKI_time']

    # Load data
    age_df = pd.read_csv(age_csv_path, dtype={'subject_id': str})
    aki_df = pd.read_csv(aki_csv_path, dtype={'subject_id': str})

    # Convert times to datetime
    age_df['admittime'] = pd.to_datetime(age_df['admittime'])
    aki_df['AKI_time'] = pd.to_datetime(aki_df['AKI_time'])

    # Merge AKI_time into age_df
    merged = age_df.merge(aki_df[['subject_id', 'AKI_time']], on='subject_id', how='left')

    # Calculate difference to 12h before AKI
    merged['delta_hours'] = ((merged['AKI_time'] - pd.Timedelta(hours=12)) - merged['admittime']).abs() / pd.Timedelta(hours=1)

    # Sort by subject_id and delta_hours, take the closest
    closest = merged.sort_values(['subject_id', 'delta_hours']).groupby('subject_id').first().reset_index()

    # Keep only subject_id and age
    result_age = closest[['subject_id', 'age']].rename(columns={'age': 'age_12h_before_AKI'})

    return result_age

# # Gebruik:
# df_age = get_age_12h_before_AKI()
# print(df_age)
