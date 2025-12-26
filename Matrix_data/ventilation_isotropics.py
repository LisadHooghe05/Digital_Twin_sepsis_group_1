# Generate df_x
from pathlib import Path
# from Matrix_data.boxplots_adam import box_plot_inotropic, box_plot_ventilation1
import numpy as np
import pandas as pd
from Tool_1.determining_AKI import AKI_detection

def extract_inotropics(item_labels, sepsis_csv, creatinine_csv, input_csvs):
    """
    item_labels: list of inotropic drugs to extract
    Returns a DataFrame with rows=patients and columns=item_labels (sum of amount in 12h before AKI)
    """
    df_id_stage = AKI_detection(sepsis_csv, creatinine_csv)
    
    if isinstance(input_csvs, str):
        input_csvs = [input_csvs]
        
    input_dfs = [pd.read_csv(f, sep=',') for f in input_csvs]
    input_df = pd.concat(input_dfs, ignore_index=True)
    input_df.columns = input_df.columns.str.strip().str.lower()
    input_df['starttime'] = pd.to_datetime(input_df['starttime'], errors='coerce')
    df_id_stage['AKI_time'] = pd.to_datetime(df_id_stage['AKI_time'], errors='coerce')

    aki_patient_ids = df_id_stage['subject_id'].unique()
    filtered_input = input_df[input_df['subject_id'].isin(aki_patient_ids)]

    patient_data = pd.DataFrame({'subject_id': aki_patient_ids})
    
    for label in item_labels:
        label_lower = label.lower()
        df_label = filtered_input[filtered_input['item_label'].str.lower() == label_lower].copy()
        # Remove duplicate rows for the same subject_id and item_label before proceeding with the time window
        # df_label = df_label.drop_duplicates(subset=['subject_id', 'item_label'])

        df_label = df_label.merge(
            df_id_stage[['subject_id', 'AKI_time']],
            on='subject_id', how='left'
        )
        df_label['time_diff'] = (df_label['AKI_time'] - df_label['starttime']).dt.total_seconds()
        df_label = df_label[(df_label['time_diff'] >= 0) & (df_label['time_diff'] <= 12*3600)]
        
        sum_df = df_label.groupby('subject_id', as_index=False)['amount'].sum()
        sum_df.rename(columns={'amount': label}, inplace=True)
        patient_data = patient_data.merge(sum_df, on='subject_id', how='left')
        
    # Fill NaNs with 0 (if patient did not receive a drug)
    #patient_data.fillna(0, inplace=True) -> leaved out so that we will get not available as value
    
    return patient_data

def extract_ventilation(item_labels, sepsis_csv, creatinine_csv, procedure_event_csv):
    """
    item_labels: list of ventilation events to extract (e.g., 'Intubation', 'Non-invasive ventilation', 'Invasive ventilation')
    Returns a DataFrame with rows=patients and columns=item_labels.
    Intubation: 1 if any intubation event overlaps the 12h window, else 0.
    Non/Invasive ventilation: minutes of ventilation within the 12h window (union of overlapping intervals).
    """

    def _union_minutes(group):
        # group must contain clip_start, clip_end (datetime)
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
                # overlapping / adjacent interval -> merge
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

    # times
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

    # normalize item_label safely (older pandas-friendly)
    if 'item_label' not in filtered_proc.columns:
        raise KeyError("procedureevents CSV mist kolom 'item_label'.")

    filtered_proc['item_label_norm'] = (
        filtered_proc['item_label'].fillna('').astype(str).str.lower().str.strip()
    )

    for label in item_labels:
        label_lower = label.lower().strip()

        df_label = filtered_proc[filtered_proc['item_label_norm'] == label_lower].copy()
        if df_label.empty:
            continue

        df_label = df_label.merge(
            df_id_stage[['subject_id', 'AKI_time']],
            on='subject_id', how='left'
        )

        # window bounds
        win_start = df_label['AKI_time'] - pd.Timedelta(hours=12)
        win_end   = df_label['AKI_time']

        # fill endtime if missing -> treat as instant at starttime
        endtime_filled = df_label['endtime'].fillna(df_label['starttime'])

        # clip intervals to the window
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
                        .reset_index(name=label)
            )

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

    # Hardcoded values
    inotropics = ['Dopamine', 'Norepinephrine', 'Epinephrine', 'Milrinone', 'Metoprolol', 'Esmolol', 'Verapamil', 'Diltiazem']
    ventilations = ['Intubation', 'Non-invasive ventilation', 'Invasive ventilation']
    
    # Paths to CSV files (hardcoded)
    from pathlib import Path
    REPO_ROOT = Path(__file__).resolve().parent.parent
    PATH_DATA = REPO_ROOT / "data"
    
    sepsis_path = PATH_DATA / "sepsis_diagnose_time.csv"
    creatinine_path = PATH_DATA / "creatinine_over_time.csv"
    procedureevents_path = PATH_DATA / "procedureevents_sepsis.csv"
    
    input_paths = [
        PATH_DATA / "inputevents_sepsis1.csv",
        PATH_DATA / "inputevents_sepsis2.csv",
        PATH_DATA / "inputevents_sepsis3.csv"
    ]
    
    # Extract inotropics data
    df_inotropic = extract_inotropics(inotropics, sepsis_path, creatinine_path, input_paths)
    # print(f"Aantal rijen in df_inotropic: {len(df_inotropic)}")

    # Extract ventilation data
    df_ventilation = extract_ventilation(ventilations, sepsis_path, creatinine_path, procedureevents_path)
    # print(f"Aantal rijen in df_ventilation: {len(df_ventilation)}")

    # Aangezien beide functies al gemergede dataframes teruggeven, hoeven we niet opnieuw te mergen.
    # We combineren ze simpelweg door de DataFrames aan elkaar te koppelen (met de `subject_id` als sleutel).
    df_x = df_inotropic.merge(df_ventilation, on='subject_id', how='left')
    # print(f"Aantal rijen in df_x na de merge: {len(df_x)}")

    # --- Load AKI subjects ---
    aki_subjects_path = "AKI_subjects.csv"
    aki_subjects = pd.read_csv(aki_subjects_path)
    aki_subjects["subject_id"] = aki_subjects["subject_id"].astype(str)

    # Zorg ervoor dat de subject_id kolom in beide dataframes hetzelfde type heeft (str)
    df_x["subject_id"] = df_x["subject_id"].astype(str)

    # Filter de DataFrame om alleen subject_id's in AKI_subjects.csv op te nemen
    df_x = df_x[df_x['subject_id'].isin(aki_subjects['subject_id'])]

    # --- Future-proof NaN handling ---
    # Intubation 0/1
    if "Intubation" in df_x.columns:
        df_x["Intubation"] = df_x["Intubation"].eq(1).astype(int)

    # Vul overige numerieke kolommen waar NaN in zit met 0
    num_cols = [col for col in df_x.columns if col not in ["subject_id", "Intubation"]]
    df_x[num_cols] = df_x[num_cols].where(df_x[num_cols].notna(), 0)
    # Optioneel: NaN-waarden invullen met 0 (als een patiënt geen inotropics of ventilatie heeft gekregen)
    #df_x.fillna(0, inplace=True)
    
    # Zorg ervoor dat 'subject_id' de eerste kolom is
    cols = ['subject_id'] + [col for col in df_x.columns if col != 'subject_id']
    df_x = df_x[cols]  # Herordeneer de kolommen

    # Reset index om ervoor te zorgen dat we geen indexkolom hebben
    df_x.reset_index(drop=True, inplace=True)

    # # Opslaan naar CSV (optioneel)
    # output_path = PATH_DATA / "complete_patient_data.csv"
    # df_x.to_csv(output_path, index=False)
    
    return df_x

