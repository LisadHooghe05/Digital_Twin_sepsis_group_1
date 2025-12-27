from pathlib import Path
import pandas as pd
from Matrix_data.open_as_df import open_as_df

REPO_ROOT = Path(__file__).resolve().parent.parent
PATH_DATA = REPO_ROOT / "data"

def antibiotica_df():
    """
    Returns a DataFrame with the given antibiotica per subject, closest to 12h before AKI onset.
    Uses internal file paths.
    """
    # The paths
    sepsis_path = PATH_DATA / "sepsis_diagnose_time.csv"
    
    input_paths = [
    PATH_DATA / "inputevents_sepsis1.csv",
    PATH_DATA / "inputevents_sepsis2.csv",
    PATH_DATA / "inputevents_sepsis3.csv"]
    
    # Different types of anitbiotics
    antibiotics = [
    "Vancomycin", "Teicoplanin", "Co-trimoxazole", "Sulfadiazine",
    "Sulfacetamide", "Gentamicin", "Tobramycin", "Amikacin",
    "Streptomycin", "Piperacillin", "Meropenem", "Cefepime",
    "Flucloxacillin", "Methicillin", "Bactrim (SMX/TMP)"]

    # Load AKI info
    aki_df = pd.read_csv(REPO_ROOT / "AKI_stage_output.csv",
                         dtype={'subject_id': str},
                         parse_dates=["AKI_time"])
    time_aki = aki_df[['subject_id', 'AKI_time']].copy()
    time_aki['window_start'] = time_aki['AKI_time'] - pd.Timedelta(hours=12)
    time_aki['window_end'] = time_aki['AKI_time']
    aki_ids = set(time_aki['subject_id'])

    # Load inputevents (filtered on AKI subject IDs)
    dfs = []
    for path in input_paths:
        df = open_as_df(path, sepsis_path)
        df['subject_id'] = df['subject_id'].astype(str)
        df = df[df['subject_id'].isin(aki_ids)]
        dfs.append(df)

    sepsis_df = pd.concat(dfs, ignore_index=True)
    sepsis_df['starttime'] = pd.to_datetime(sepsis_df['starttime'])
    sepsis_df['endtime'] = pd.to_datetime(sepsis_df['endtime'])

    # Merge with AKI windows
    df = sepsis_df.merge(time_aki, on='subject_id', how='left')

    # Select events in 12h window
    mask = (df['endtime'] >= df['window_start']) & (df['starttime'] <= df['window_end'])
    df_window = df.loc[mask]
    df_window = df_window[df_window['item_label'].isin(antibiotics)]

    # Binary marker
    df_window['value'] = 1

    # Pivot to binary antibiotic matrix
    ab_pivot = (
    df_window.groupby(['subject_id', 'item_label'])['value']
        .max()
        .unstack(fill_value=0)
    ).reindex(columns=antibiotics,fill_value=0)

    # Filter subject_ids
    mfa = pd.read_csv(REPO_ROOT / "AKI_subjects.csv", dtype={'subject_id': str})
    valid_ids = set(mfa['subject_id'])
    ab_pivot = ab_pivot.reindex(index=list(valid_ids), fill_value=0)

    # Make a column of the subject_ids
    ab_pivot.reset_index(inplace=True)

    return ab_pivot

# To run the code
#result = antibiotica_df(sepsis_path, input_paths, antibiotics)
#print(result.head())

def other_meds_df():
    """
    Returns a DataFrame with the other types of given meds per subject, closest to 12h before AKI onset.
    Uses internal file paths.
    """
    # The paths
    sepsis_path = PATH_DATA / "sepsis_diagnose_time.csv"
    
    input_paths = [
        PATH_DATA / "inputevents_sepsis1.csv",
        PATH_DATA / "inputevents_sepsis2.csv",
        PATH_DATA / "inputevents_sepsis3.csv"
    ]

    med_list = [
        "Cyclosporine",
        "Ketorolac (Toradol)"
    ]

    # Load AKI information
    aki_df = pd.read_csv(
        REPO_ROOT / "AKI_stage_output.csv",
        dtype={'subject_id': str},
        parse_dates=["AKI_time"]
    )

    time_aki = aki_df[['subject_id', 'AKI_time']].copy()
    time_aki['window_start'] = time_aki['AKI_time'] - pd.Timedelta(hours=12)
    time_aki['window_end']   = time_aki['AKI_time']
    aki_ids = set(time_aki['subject_id'])

    # Load sepsis AKI subject_ids
    mfa_df = pd.read_csv(REPO_ROOT / "AKI_subjects.csv", dtype={'subject_id': str})
    mfa_ids = set(mfa_df['subject_id'])

    # Only keep IDs that appear in BOTH AKI and MFA
    final_ids = aki_ids.intersection(mfa_ids)

    # Load inputevents (filtered on selected IDs) 
    dfs = []
    for path in input_paths:
        df = open_as_df(path, sepsis_path)
        df['subject_id'] = df['subject_id'].astype(str)
        df = df[df['subject_id'].isin(final_ids)]
        dfs.append(df)

    sepsis_df = pd.concat(dfs, ignore_index=True)
    sepsis_df['starttime'] = pd.to_datetime(sepsis_df['starttime'])
    sepsis_df['endtime']   = pd.to_datetime(sepsis_df['endtime'])

    # Merge with AKI windows
    df = sepsis_df.merge(time_aki, on='subject_id', how='left')

    # Select 12h window before AKI 
    mask = (
        (df['endtime']   >= df['window_start']) &
        (df['starttime'] <= df['window_end']))
    df_window = df.loc[mask]

    # Filter selected medication 
    df_window = df_window[df_window['item_label'].isin(med_list)]

    # Sum mg/ml per medication
    med_pivot = (
        df_window.groupby(['subject_id', 'item_label'])['totalamount']
        .sum()
        .unstack())

    med_pivot = med_pivot.reindex(columns=med_list)

    # Include ALL subject_ids that should remain
    med_pivot = med_pivot.reindex(index=list(final_ids))

    # Make a column of the subject_ids instead of an index
    med_pivot.reset_index(inplace=True)

    # Save the med_pivot matrix
    #out = REPO_ROOT / "other_meds_window_12h_sum.csv"
    #med_pivot.to_csv(out, index=False)

    return med_pivot


# To run the code
#other_result = other_meds_df(sepsis_path, input_paths, other_meds)
#print(other_result.head())