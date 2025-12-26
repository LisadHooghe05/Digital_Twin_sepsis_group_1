import pandas as pd
from pathlib import Path

def get_vasopressor_matrix_12h():
    """
    Builds a wide matrix of vasopressor doses per subject within 12h before AKI onset.
    Rows: subject_id
    Columns: vasopressor types, values: summed dose amounts (within the 12h window)
    Returns a pandas DataFrame.
    """

    # Paths
    REPO_ROOT = Path(__file__).resolve().parent.parent
    PATH_DATA = REPO_ROOT / "data"

    # Files (zelfde als origineel)
    vaso_file = PATH_DATA / "vasopressors.csv"
    vaso_info_file = PATH_DATA / "VASO.csv"
    aki_file = "AKI_stage_output.csv"
    aki_stage_file = "AKI_subjects.csv"

    # Read data (zelfde als origineel)
    vaso_df = pd.read_csv(vaso_file, dtype={'stay_id': str})
    vaso_info_df = pd.read_csv(vaso_info_file, dtype={'stay_id': str, 'subject_id': str})
    aki_df = pd.read_csv(aki_file, dtype={'subject_id': str})
    aki_stage_df = pd.read_csv(aki_stage_file, dtype={'subject_id': str})

    # --- Fix 1: subject_id koppelen via stay_id zonder duplicatie ---
    stay_subject = vaso_info_df[['stay_id', 'subject_id']].drop_duplicates(subset=['stay_id'])
    vaso_merged = pd.merge(
        vaso_df,
        stay_subject,
        on='stay_id',
        how='left'
    )

    # --- Fix 2: vasopressor_type koppelen via drugnaam (label ↔ drug) zonder duplicatie ---
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
        how='left'
    )

    # Convert times to datetime (zelfde stijl)
    vaso_merged['starttime'] = pd.to_datetime(vaso_merged['starttime'], errors='coerce')
    vaso_merged['endtime']   = pd.to_datetime(vaso_merged['endtime'], errors='coerce')
    aki_df['AKI_time'] = pd.to_datetime(aki_df['AKI_time'], errors='coerce')

    # (optioneel maar veilig) als er meerdere AKI_times per subject zijn: pak earliest
    aki_df = aki_df.sort_values('AKI_time').drop_duplicates(subset=['subject_id'], keep='first')
    aki_stage_df = aki_stage_df.drop_duplicates(subset=['subject_id'], keep='first')

    # Merge AKI time and stage (zelfde als origineel)
    vaso_merged = pd.merge(
        vaso_merged,
        aki_df[['subject_id','AKI_time']],
        on='subject_id',
        how='left'
    )
    vaso_merged = pd.merge(
        vaso_merged,
        aki_stage_df[['subject_id','AKI_stage']],
        on='subject_id',
        how='left'
    )

    # Zorg dat amount numeriek is
    vaso_merged['amount'] = pd.to_numeric(vaso_merged['amount'], errors='coerce')

    # Drop rows met missings die we echt nodig hebben
    vaso_merged = vaso_merged.dropna(subset=[
        'subject_id','vasopressor_type','amount','AKI_stage',
        'starttime','endtime','AKI_time'
    ])

    # --- Fix 3: filter op overlap met [AKI_time-12h, AKI_time] ---
    win_start = vaso_merged['AKI_time'] - pd.Timedelta(hours=12)
    win_end   = vaso_merged['AKI_time']

    overlap_start = pd.concat([vaso_merged['starttime'], win_start], axis=1).max(axis=1)
    overlap_end   = pd.concat([vaso_merged['endtime'],   win_end],   axis=1).min(axis=1)

    overlap_sec = (overlap_end - overlap_start).dt.total_seconds()
    mask = overlap_sec > 0
    vaso_12h = vaso_merged.loc[mask].copy()
    vaso_12h['overlap_sec'] = overlap_sec.loc[mask].values

    # --- Fix 4: alleen het deel van amount binnen het venster meetellen (proportioneel) ---
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






