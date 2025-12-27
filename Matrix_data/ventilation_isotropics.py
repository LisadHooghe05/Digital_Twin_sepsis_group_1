from pathlib import Path
import numpy as np
import pandas as pd
from Tool_1.determining_AKI import AKI_detection

def extract_inotropics(item_labels, sepsis_csv, creatinine_csv, input_csvs):
    """
    Extracts the total administered inotropic drug amounts for AKI patients 
    within 12 hours before AKI onset and returns a wide-format patient-level matrix.

    Parameters
    - item_labels : list of str -> Names of inotropic drugs to extract.
    - sepsis_csv : str -> Path to the sepsis diagnosis CSV file.
    - creatinine_csv : str -> Path to the creatinine measurements CSV file used for AKI detection.
    - input_csvs : str or list of str -> Path(s) to the input events CSV file(s) containing administered drugs.

    Returns
    - pandas.DataFrame
        A DataFrame where rows correspond to patients (subject_id) and columns 
        correspond to the specified inotropic drugs. Values are the summed 
        amounts administered within the 12 hours before AKI onset. Patients 
        without administration of a drug will have NaN for that column.
    """
    # Call function to get the AKI patients and their AKI time
    df_id_stage = AKI_detection(sepsis_csv, creatinine_csv)
    
    # Ensure input_csvs is a list
    if isinstance(input_csvs, str):
        input_csvs = [input_csvs]
        
    # Load and concatenate input events CSVs
    input_dfs = [pd.read_csv(f, sep=',') for f in input_csvs]
    input_df = pd.concat(input_dfs, ignore_index=True)
    
    # Standardize column names and parse starttime
    input_df.columns = input_df.columns.str.strip().str.lower()
    input_df['starttime'] = pd.to_datetime(input_df['starttime'], errors='coerce')
    df_id_stage['AKI_time'] = pd.to_datetime(df_id_stage['AKI_time'], errors='coerce')

    # Filter only AKI patients
    aki_patient_ids = df_id_stage['subject_id'].unique()
    filtered_input = input_df[input_df['subject_id'].isin(aki_patient_ids)]

    # Initialize output DataFrame with all AKI patients
    patient_data = pd.DataFrame({'subject_id': aki_patient_ids})
    
    # Loop through each inotropic drug
    for label in item_labels:
        label_lower = label.lower()
        # Merge AKI time to calculate time difference
        df_label = filtered_input[filtered_input['item_label'].str.lower() == label_lower].copy()
        df_label = df_label.merge(df_id_stage[['subject_id', 'AKI_time']],on='subject_id', how='left')
        # Keep only administrations within 12h before AKI onset
        df_label['time_diff'] = (df_label['AKI_time'] - df_label['starttime']).dt.total_seconds()
        df_label = df_label[(df_label['time_diff'] >= 0) & (df_label['time_diff'] <= 12*3600)]
        # Sum amounts per patient
        sum_df = df_label.groupby('subject_id', as_index=False)['amount'].sum()
        sum_df.rename(columns={'amount': label}, inplace=True)
        # Merge with final patient-level matrix
        patient_data = patient_data.merge(sum_df, on='subject_id', how='left')
        
    return patient_data

def extract_ventilation(item_labels, sepsis_csv, creatinine_csv, procedure_event_csv):
    """
    Extract ventilation events for AKI patients within 12 hours before AKI onset 
    and return a patient-level matrix.

    The function processes procedure events to capture intubation and ventilation:
    - 'Intubation': 1 if any intubation event overlaps the 12-hour window before AKI, else 0.
    - 'Non-invasive ventilation' / 'Invasive ventilation': total minutes of ventilation 
      within the 12-hour window, computed as the union of overlapping intervals.

    Parameters
    - item_labels : list of str -> Ventilation event types to extract (e.g., 'Intubation', 'Non-invasive ventilation').
    - sepsis_csv : str -> Path to the sepsis diagnosis CSV file.
    - creatinine_csv : str -> Path to the creatinine measurements CSV file used for AKI detection.
    - procedure_event_csv : str -> Path to the procedure events CSV file.

    Returns
    - pandas.DataFrame
        A DataFrame where rows represent patients (subject_id) and columns represent 
        the specified ventilation events. Values indicate 1 for intubation if present, 
        or total minutes of ventilation within the 12-hour window for other events.
    """

    def _union_minutes(group):
        # Group must contain clip_start, clip_end (datetime)
        g = group[['clip_start', 'clip_end']].sort_values('clip_start')
        total_min = 0.0
        cur_s = None
        cur_e = None

        for s, e in zip(g['clip_start'].values, g['clip_end'].values):
            if pd.isna(s) or pd.isna(e):
                continue
            if cur_s is None:
                cur_s, cur_e = s, e
            else:
                # Overlapping / adjacent interval -> merge
                if s <= cur_e:
                    if e > cur_e:
                        cur_e = e
                else:
                    total_min += (cur_e - cur_s) / np.timedelta64(1, 'm')
                    cur_s, cur_e = s, e

        if cur_s is not None:
            total_min += (cur_e - cur_s) / np.timedelta64(1, 'm')
        # print(total_min)
        return total_min

    df_id_stage = AKI_detection(sepsis_csv, creatinine_csv)

    proc_df = pd.read_csv(procedure_event_csv)
    proc_df.columns = proc_df.columns.str.strip().str.lower()

    # Convert starttime, endtime and AKI_time to datetimes
    proc_df['starttime'] = pd.to_datetime(proc_df['starttime'], errors='coerce')
    if 'endtime' in proc_df.columns:
        proc_df['endtime'] = pd.to_datetime(proc_df['endtime'], errors='coerce')
    elif 'stoptime' in proc_df.columns:
        proc_df['endtime'] = pd.to_datetime(proc_df['stoptime'], errors='coerce')
    else:
        proc_df['endtime'] = pd.NaT  # fallback

    df_id_stage['AKI_time'] = pd.to_datetime(df_id_stage['AKI_time'], errors='coerce')

    aki_patient_ids = df_id_stage['subject_id'].unique()
    filtered_proc = proc_df[proc_df['subject_id'].isin(aki_patient_ids)].copy()

    patient_data = pd.DataFrame({'subject_id': aki_patient_ids})

    # Normalize item_label safely 
    if 'item_label' not in filtered_proc.columns:
        raise KeyError("procedureevents CSV mist kolom 'item_label'.")

    filtered_proc['item_label_norm'] = (filtered_proc['item_label'].fillna('').astype(str).str.lower().str.strip())

    for label in item_labels:
        label_lower = label.lower().strip()

        df_label = filtered_proc[filtered_proc['item_label_norm'] == label_lower].copy()
        if df_label.empty:
            continue

        df_label = df_label.merge(df_id_stage[['subject_id', 'AKI_time']],on='subject_id', how='left')

        # Window bounds
        win_start = df_label['AKI_time'] - pd.Timedelta(hours=12)
        win_end   = df_label['AKI_time']

        # Fill endtime if missing -> treat as instant at starttime
        endtime_filled = df_label['endtime'].fillna(df_label['starttime'])

        # Clip intervals to the window
        clip_start = pd.concat([df_label['starttime'], win_start], axis=1).max(axis=1)
        clip_end   = pd.concat([endtime_filled,        win_end],   axis=1).min(axis=1)

        overlap_sec = (clip_end - clip_start).dt.total_seconds()
        mask = overlap_sec > 0
        df_label = df_label.loc[mask].copy()
        if df_label.empty:
            continue

        df_label['clip_start'] = clip_start.loc[mask].values
        df_label['clip_end']   = clip_end.loc[mask].values

        if label_lower == 'intubation':
            # any overlap => 1
            df_label_agg = df_label.groupby('subject_id', as_index=False).size()
            df_label_agg[label] = 1
            df_label_agg = df_label_agg[['subject_id', label]]
        else:
            # UNION of intervals => minutes within 12h window (<= 720)
            df_label_agg = (
                df_label.groupby('subject_id')
                        .apply(_union_minutes,include_groups=False)
                        .reset_index(name=label))

        patient_data = patient_data.merge(df_label_agg, on='subject_id', how='left')

    # Make intubation binary 0/1
    if "Intubation" in patient_data.columns:
        patient_data["Intubation"] = patient_data["Intubation"].eq(1).astype(int)
    
    return patient_data

def build_patient_feature_matrix():
    """
    Extracts both inotropics and ventilation information and combines them into a single DataFrame.
    
    No parameters are passed to this function. All the necessary values are hardcoded inside the function.
    
    Returns:
    - A DataFrame with rows=patients and columns including inotropics and ventilation data.
    """

    # Wanted values
    inotropics = ['Dopamine', 'Norepinephrine', 'Epinephrine', 'Milrinone', 'Metoprolol', 'Esmolol', 'Verapamil', 'Diltiazem']
    ventilations = ['Intubation', 'Non-invasive ventilation', 'Invasive ventilation']
    
    # Paths to CSV files
    from pathlib import Path
    REPO_ROOT = Path(__file__).resolve().parent.parent
    PATH_DATA = REPO_ROOT / "data"
    
    sepsis_path = PATH_DATA / "sepsis_diagnose_time.csv"
    creatinine_path = PATH_DATA / "creatinine_over_time.csv"
    procedureevents_path = PATH_DATA / "procedureevents_sepsis.csv"
    
    input_paths = [
        PATH_DATA / "inputevents_sepsis1.csv",
        PATH_DATA / "inputevents_sepsis2.csv",
        PATH_DATA / "inputevents_sepsis3.csv"]
    
    # Extract inotropics data
    df_inotropic = extract_inotropics(inotropics, sepsis_path, creatinine_path, input_paths)

    # Extract ventilation data
    df_ventilation = extract_ventilation(ventilations, sepsis_path, creatinine_path, procedureevents_path)

    # Combine both dataframes by using the subject_ids
    df_x = df_inotropic.merge(df_ventilation, on='subject_id', how='left')

    # Load AKI subjects
    aki_subjects_path = "AKI_subjects.csv"
    aki_subjects = pd.read_csv(aki_subjects_path)
    aki_subjects["subject_id"] = aki_subjects["subject_id"].astype(str)

    # Ensure that the 'subject_id' column in both DataFrames has the same type (str)
    df_x["subject_id"] = df_x["subject_id"].astype(str)

    # Filter the DataFrame to include only subject_ids present in AKI_subjects.csv
    df_x = df_x[df_x['subject_id'].isin(aki_subjects['subject_id'])]

    # Intubation 0/1
    if "Intubation" in df_x.columns:
        df_x["Intubation"] = df_x["Intubation"].eq(1).astype(int)

    # Fill remaining numeric columns with 0 where values are NaN
    num_cols = [col for col in df_x.columns if col not in ["subject_id", "Intubation"]]
    df_x[num_cols] = df_x[num_cols].where(df_x[num_cols].notna(), 0)
    
    # Ensure that 'subject_id' is the first column
    cols = ['subject_id'] + [col for col in df_x.columns if col != 'subject_id']
    df_x = df_x[cols]  # Herordeneer de kolommen

    # Reset index to ensure there is no separate index column
    df_x.reset_index(drop=True, inplace=True)

    # # Save to CSV
    # output_path = PATH_DATA / "complete_patient_data.csv"
    # df_x.to_csv(output_path, index=False)
    
    return df_x

