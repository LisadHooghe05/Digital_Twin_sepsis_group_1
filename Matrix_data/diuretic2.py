import pandas as pd
from pathlib import Path
import sys
# REPO_ROOT = Path(__file__).resolve().parent.parent
# sys.path.insert(0,str(REPO_ROOT))

def get_diuretics2_matrix_12h():
    """
    Builds a 0/1 matrix of diuretics administration within 12h before AKI onset.
    Uses default CSV file paths inside the 'data' folder relative to this script.

    Returns:
    - pd.DataFrame: subjects x diuretic types (0/1)
    """
    # Paths
    REPO_ROOT = Path(__file__).resolve().parent.parent
    PATH_DATA = REPO_ROOT / "data"

    diu_file = PATH_DATA / "DIU.csv"
    aki_file = "AKI_subjects.csv"
    aki_onset_file = "AKI_stage_output.csv"

    # Read CSVs
    diu_df = pd.read_csv(diu_file, dtype={'subject_id': str})
    aki_subject_df = pd.read_csv(aki_file, dtype={'subject_id': str})
    aki_onset_df = pd.read_csv(aki_onset_file, dtype={'subject_id': str})
    aki_subjects = set(aki_subject_df['subject_id'])

    # Keep only AKI subjects
    diu_df = diu_df[diu_df['subject_id'].isin(aki_subjects)]

    # Convert times to datetime
    diu_df['stoptime'] = pd.to_datetime(diu_df['stoptime'])
    aki_onset_df['AKI_time'] = pd.to_datetime(aki_onset_df['AKI_time'])

    # Merge to get AKI onset time
    diu_df = diu_df.merge(
        aki_onset_df[['subject_id','AKI_time']],
        on='subject_id',
        how='left')

    # Keep only drugs within 12h before AKI onset
    diu_df['hours_before_AKI'] = (diu_df['AKI_time'] - diu_df['stoptime']).dt.total_seconds() / 3600
    diu_df_12h = diu_df[(diu_df['hours_before_AKI'] >= 0) & (diu_df['hours_before_AKI'] <= 12)].copy()

    # Convert to 0/1 matrix: subjects x diuretic types
    diu2_matrix_12h = diu_df_12h.pivot_table(
        index='subject_id',
        columns='drug',
        values='stoptime',
        aggfunc='count')

    # Convert counts >0 to 1
    diu2_matrix_12h = (diu2_matrix_12h > 0).astype(int).reset_index()
    diu2_matrix_12h = diu2_matrix_12h.round(2)

    return diu2_matrix_12h

