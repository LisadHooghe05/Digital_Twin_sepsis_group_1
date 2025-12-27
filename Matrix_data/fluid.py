import pandas as pd
from pathlib import Path
from Matrix_data.open_as_df import open_as_df

def get_fluid_matrix_12h():
    """
    Create a patient-level matrix of total fluid intake within 12 hours before AKI onset.

    Returns
    - pandas.DataFrame
        Rows represent patients and columns represent fluid types, with values
        indicating total administered volume in the 12 hours prior to AKI.
    """
    # The Paths
    REPO_ROOT = Path(__file__).resolve().parent.parent
    PATH_DATA = REPO_ROOT / "data"

    input_paths = [
        PATH_DATA / "inputevents_sepsis1.csv",
        PATH_DATA / "inputevents_sepsis2.csv",
        PATH_DATA / "inputevents_sepsis3.csv"]

    sepsis_path = PATH_DATA / "sepsis_diagnose_time.csv"

    # Load sepsis data
    sepsis_data = pd.concat(
        [open_as_df(path, sepsis_path) for path in input_paths],
        ignore_index=True)

    # Filter fluids
    fluids_subset = sepsis_data[sepsis_data['item_category'] == 'Fluids/Intake'][[
        'subject_id','item_label','amount','rate','starttime']].copy()

    fluids_subset['subject_id'] = fluids_subset['subject_id'].astype(str)
    fluids_subset['starttime'] = pd.to_datetime(fluids_subset['starttime'], errors='coerce')

    # Load AKI information
    AKI_df = pd.read_csv("AKI_subjects.csv")
    AKI_df['subject_id'] = AKI_df['subject_id'].astype(str)

    aki_onset_df = pd.read_csv("AKI_stage_output.csv")
    aki_onset_df['subject_id'] = aki_onset_df['subject_id'].astype(str)
    aki_onset_df['AKI_time'] = pd.to_datetime(aki_onset_df['AKI_time'], errors='coerce', dayfirst=True)
    aki_onset_df = aki_onset_df.drop_duplicates(subset=['AKI_time'])

    # Merge aubject_id with AKI_time and AKI_stage
    merged = (fluids_subset
        .merge(aki_onset_df[['subject_id','AKI_time']], on='subject_id', how='inner')
        .merge(AKI_df[['subject_id','AKI_stage']], on='subject_id', how='inner'))

    # Compute hours_before_onset
    merged['hours_before_onset'] = (
        merged['AKI_time'] - merged['starttime']
    ).dt.total_seconds() / 3600

    # Look at which values are between 12 hours before AKI_onset and AKI_onset
    merged_12h = merged[(merged['hours_before_onset'] >= 0) &
                        (merged['hours_before_onset'] <= 12)]

    # Aggregate total per patient
    total_per_patient = merged_12h.groupby(
        ['subject_id','item_label','AKI_stage'])['amount'].sum().reset_index()

    total_per_patient.rename(columns={'amount':'total_fluid_12h'}, inplace=True)

    # Build matrix
    sepsis_time = pd.read_csv(PATH_DATA / "sepsis_diagnose_time.csv")
    sepsis_time['subject_id'] = sepsis_time['subject_id'].astype(str)

    all_subjects = sepsis_time[['subject_id']].drop_duplicates()

    fluid_matrix_12h = (
        total_per_patient
        .pivot_table(
            index='subject_id',
            columns='item_label',
            values='total_fluid_12h',
            aggfunc='sum'
        ).reset_index())
    
    fluid_matrix_12h = all_subjects.merge(fluid_matrix_12h, on='subject_id', how='left')
    fluid_matrix_12h = fluid_matrix_12h.round(2)

    return fluid_matrix_12h
