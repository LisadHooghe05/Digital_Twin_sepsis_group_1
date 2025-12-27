import pandas as pd
from pathlib import Path

def get_vasopressor_matrix_12h():
    """
    Build a wide-format matrix of vasopressor doses for sepsis patients within 12 hours before AKI onset.

    The function performs the following steps:
    - Loads vasopressor administration data and AKI onset times.
    - Links subject IDs via stay_id and maps vasopressor types.
    - Filters events to the 12-hour window before AKI onset.
    - Calculates the proportion of each dose that falls within the window.
    - Aggregates total dose per vasopressor type per patient.

    Returns
    - pandas.DataFrame
        A wide-format DataFrame where rows represent patients (subject_id), 
        columns represent vasopressor types, and values are the summed doses 
        administered within the 12-hour window before AKI onset, rounded to 2 decimals.
    """

    # Paths
    REPO_ROOT = Path(__file__).resolve().parent.parent
    PATH_DATA = REPO_ROOT / "data"

    # Loading AKI and vasopressor data
    vaso_file = PATH_DATA / "vasopressors.csv"
    vaso_info_file = PATH_DATA / "VASO.csv"
    aki_file = "AKI_stage_output.csv"
    aki_stage_file = "AKI_subjects.csv"

    # Read data 
    vaso_df = pd.read_csv(vaso_file, dtype={'stay_id': str})
    vaso_info_df = pd.read_csv(vaso_info_file, dtype={'stay_id': str, 'subject_id': str})
    aki_df = pd.read_csv(aki_file, dtype={'subject_id': str})
    aki_stage_df = pd.read_csv(aki_stage_file, dtype={'subject_id': str})

    # Link subject_id via stay_id without duplication
    stay_subject = vaso_info_df[['stay_id', 'subject_id']].drop_duplicates(subset=['stay_id'])
    vaso_merged = pd.merge(
        vaso_df,
        stay_subject,
        on='stay_id',
        how='left')

    # Link vasopressor_type via drug name (label and drug) without duplication
    drug_map = (vaso_info_df[['drug', 'vasopressor_type']]
                .dropna()
                .assign(drug_norm=lambda d: d['drug'].astype(str).str.lower().str.strip())
                .drop_duplicates(subset=['drug_norm'], keep='first'))

    vaso_merged['label_norm'] = vaso_merged['label'].astype(str).str.lower().str.strip()

    vaso_merged = pd.merge(
        vaso_merged,
        drug_map[['drug_norm', 'vasopressor_type']],
        left_on='label_norm',
        right_on='drug_norm',
        how='left')

    # Convert times to datetime
    vaso_merged['starttime'] = pd.to_datetime(vaso_merged['starttime'], errors='coerce')
    vaso_merged['endtime']   = pd.to_datetime(vaso_merged['endtime'], errors='coerce')
    aki_df['AKI_time'] = pd.to_datetime(aki_df['AKI_time'], errors='coerce')

    # If there are multiple AKI_times per subject -> take the earliest
    aki_df = aki_df.sort_values('AKI_time').drop_duplicates(subset=['subject_id'], keep='first')
    aki_stage_df = aki_stage_df.drop_duplicates(subset=['subject_id'], keep='first')

    # Merge AKI time and stage 
    vaso_merged = pd.merge(
        vaso_merged,
        aki_df[['subject_id','AKI_time']],
        on='subject_id',
        how='left')
    
    vaso_merged = pd.merge(
        vaso_merged,
        aki_stage_df[['subject_id','AKI_stage']],
        on='subject_id',
        how='left')

    # Ensure that 'amount' is numeric
    vaso_merged['amount'] = pd.to_numeric(vaso_merged['amount'], errors='coerce')

    # Remove rows with missing data in columns that are essential for analysis
    vaso_merged = vaso_merged.dropna(subset=[
        'subject_id','vasopressor_type','amount','AKI_stage',
        'starttime','endtime','AKI_time'])

    # Filter for overlap with [AKI_time - 12h, AKI_time]
    win_start = vaso_merged['AKI_time'] - pd.Timedelta(hours=12)
    win_end   = vaso_merged['AKI_time']

    overlap_start = pd.concat([vaso_merged['starttime'], win_start], axis=1).max(axis=1)
    overlap_end   = pd.concat([vaso_merged['endtime'],   win_end],   axis=1).min(axis=1)

    overlap_sec = (overlap_end - overlap_start).dt.total_seconds()
    mask = overlap_sec > 0
    vaso_12h = vaso_merged.loc[mask].copy()
    vaso_12h['overlap_sec'] = overlap_sec.loc[mask].values

    # Include only the part of 'amount' that falls within the window, proportionally 
    total_sec = (vaso_12h['endtime'] - vaso_12h['starttime']).dt.total_seconds().clip(lower=1)
    frac = (vaso_12h['overlap_sec'] / total_sec).clip(lower=0, upper=1)
    vaso_12h['amount_in_window'] = vaso_12h['amount'] * frac

    # Create wide matrix: subjects x vasopressors, summed dose
    vaso_matrix = vaso_12h.pivot_table(
        index='subject_id',
        columns='vasopressor_type',
        values='amount_in_window',
        aggfunc='sum'
    ).reset_index()

    return vaso_matrix.round(2)






