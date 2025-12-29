import pandas as pd
from pathlib import Path

def get_vasopressor_matrix_12h():
    """
    Create a patient-level matrix of vasopressor doses within 12 hours before AKI onset.

    - Links vasopressor events to `subject_id` via `stay_id`.
    - Filters events overlapping the 12-hour window before AKI onset.
    - Aggregates total `amount` per patient per vasopressor type.

    Returns:
    - pd.DataFrame: rows = subject_id, columns = vasopressor types,
      values = summed dose within 12 hours (rounded to 2 decimals).
    """
    REPO_ROOT = Path(__file__).resolve().parent.parent
    PATH_DATA = REPO_ROOT / "data"

    vaso_file = PATH_DATA / "VASO.csv"
    events_file = PATH_DATA / "vasopressors.csv"
    aki_file = "AKI_stage_output.csv"

    # Load data
    vaso = pd.read_csv(vaso_file, dtype=str)
    events = pd.read_csv(events_file, dtype=str)
    aki = pd.read_csv(aki_file, dtype=str)

    # Strip spaces and unify column names
    vaso.columns = vaso.columns.str.strip().str.lower()
    events.columns = events.columns.str.strip().str.lower()
    aki.columns = aki.columns.str.strip().str.lower()

    # Convert times
    events['starttime'] = pd.to_datetime(events['starttime'], errors='coerce')
    events['endtime']   = pd.to_datetime(events['endtime'], errors='coerce')
    aki['aki_time'] = pd.to_datetime(aki['aki_time'], errors='coerce')

    # Keep only subject_id and stay_id in vaso
    vaso = vaso[['subject_id', 'stay_id']].drop_duplicates()

    # Merge subject_id into vasopressor events
    events = events.merge(vaso, on='stay_id', how='inner')

    # Merge AKI time
    events = events.merge(aki[['subject_id','aki_time']], on='subject_id', how='inner')

    # Compute 12h window
    events['win_start'] = events['aki_time'] - pd.Timedelta(hours=12)
    mask = (events['starttime'] < events['aki_time']) & (events['endtime'] > events['win_start'])
    df_12h = events.loc[mask].copy()

    # Compute overlap fraction
    df_12h['amount'] = pd.to_numeric(df_12h['amount'], errors='coerce')
    df_12h = df_12h.dropna(subset=['amount'])
    overlap_start = df_12h[['starttime','win_start']].max(axis=1)
    overlap_end   = df_12h[['endtime','aki_time']].min(axis=1)
    df_12h['overlap_sec'] = (overlap_end - overlap_start).dt.total_seconds()
    total_sec = (df_12h['endtime'] - df_12h['starttime']).dt.total_seconds().clip(lower=1)
    df_12h['amount_in_window'] = df_12h['amount'] * (df_12h['overlap_sec'] / total_sec)

    # Normalize labels
    df_12h['drug'] = df_12h['label'].astype(str).str.strip().str.title()

    # Aggregate per subject_id x drug
    vaso_matrix = (
        df_12h.groupby(['subject_id','drug'])['amount_in_window']
        .sum()
        .unstack(fill_value=0)
        .reset_index())
    
    # Add 'vaso_' prefix to all the columns
    vaso_matrix = vaso_matrix.rename(columns=lambda x: f"vaso_{x}" if x != 'subject_id' else x)

    vaso_matrix = vaso_matrix.round(2)
    return vaso_matrix

# # Usage
# t = get_vasopressor_matrix_12h()
# t.to_csv("vaso_matrix_12h.csv", index=False)
# print(t)
