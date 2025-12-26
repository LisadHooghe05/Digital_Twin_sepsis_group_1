import pandas as pd
from pathlib import Path
from functools import reduce
import numpy as np

def get_vitals_matrix_12h():
    """
    Build a wide vitals matrix (subject_id as rows, vitals as columns)
    for measurements within 12 hours before AKI onset.

    Returns:
        pd.DataFrame: vitals matrix per subject
    """

    # Paths
    PATH_DATA = Path(__file__).resolve().parent.parent / "data"

    vitals_files = [PATH_DATA/"vitals1.csv", PATH_DATA/"vitals2.csv", PATH_DATA/"vitals3.csv"]
    sepsis_file = PATH_DATA/"sepsis_diagnose_time.csv"
    aki_file = "AKI_subjects.csv"
    aki_times_file = "AKI_stage_output.csv"
    sepsis_df = pd.read_csv(sepsis_file, dtype={'stay_id': str, 'subject_id': str})

#     # Function to load and enrich a vitals file
#     def load_vitals_with_subject_id(file_path):
#             df = pd.read_csv(file_path, dtype={'stay_id': str})
#             df = df.merge(sepsis_df[['stay_id', 'subject_id']], on='stay_id', how='left')
#             return df

#     # Load all 3 vitals files
#     v1 = load_vitals_with_subject_id(PATH_DATA/"vitals1.csv")
#     v2 = load_vitals_with_subject_id(PATH_DATA/"vitals2.csv")
#     v3 = load_vitals_with_subject_id(PATH_DATA/"vitals3.csv")

#     # Count unique subject_id’s per file
#     u1 = v1['subject_id'].nunique()
#     u2 = v2['subject_id'].nunique()
#     u3 = v3['subject_id'].nunique()

#     # Combined unique subject_id’s across all files
#     combined_unique = pd.concat([v1['subject_id'], v2['subject_id'], v3['subject_id']]).nunique()

#     # Print results
#     print(f"Unieke subject_id's in vitals1.csv: {u1}")
#     print(f"Unieke subject_id's in vitals2.csv: {u2}")
#     print(f"Unieke subject_id's in vitals3.csv: {u3}")
#     print(f"Totaal unieke subject_id's over alle 3 bestanden: {combined_unique}")

    # Read sepsis mapping and AKI subjects
    aki_stage_df = pd.read_csv(aki_file, dtype={'subject_id': str})
    aki_subjects = set(aki_stage_df['subject_id'])

    # Read AKI onset times
    aki_times_df = pd.read_csv(aki_times_file, dtype={'subject_id': str})
    aki_times_df['AKI_time'] = pd.to_datetime(aki_times_df['AKI_time'], errors='coerce')
    aantal_zonder_AKI_time = aki_times_df['AKI_time'].isna().sum()
    # print("Aantal subject_id’s zonder AKI_time:", aantal_zonder_AKI_time)
    # add all vital data
    vitals_list = [pd.read_csv(vf, dtype={'stay_id': str}) for vf in vitals_files]
    vitals_df = pd.concat(vitals_list, ignore_index=True)

    # add subject id's couple them to the stay_id
    vitals_df = vitals_df.merge(sepsis_df[['stay_id', 'subject_id']], on='stay_id', how='left')
    # only AKI subjects
    vitals_df = vitals_df[vitals_df['subject_id'].isin(aki_subjects)]

    # change charttime
    vitals_df['charttime'] = pd.to_datetime(vitals_df['charttime'], errors='coerce')

    # add AKI_time
    vitals_df = vitals_df.merge(aki_times_df[['subject_id', 'AKI_time']], on='subject_id', how='left')

    # compute hours hours_before_AKI and filter on 12h
    vitals_df['hours_before_AKI'] = (vitals_df['AKI_time'] - vitals_df['charttime']).dt.total_seconds() / 3600
    vitals_df_12h = vitals_df[(vitals_df['hours_before_AKI'] >= 0) & (vitals_df['hours_before_AKI'] <= 12)]

    # filter extreme values per vital
    vitals_df_12h = vitals_df_12h[
            ((vitals_df_12h['label'] == 'Heart Rate') & vitals_df_12h['valuenum'].between(20, 300)) |
            ((vitals_df_12h['label'] == 'Systolic Blood Pressure') & vitals_df_12h['valuenum'].between(50, 250)) |
            ((vitals_df_12h['label'] == 'Diastolic Blood Pressure') & vitals_df_12h['valuenum'].between(30, 150)) |
            ((vitals_df_12h['label'] == 'Mean Arterial Pressure') & vitals_df_12h['valuenum'].between(30, 200)) |
            ((vitals_df_12h['label'] == 'Temperature') & vitals_df_12h['valuenum'].between(30, 45)) |
            ((vitals_df_12h['label'] == 'Respiratory Rate') & vitals_df_12h['valuenum'].between(5, 60)) |
            ((vitals_df_12h['label'] == 'Oxygen Saturation') & vitals_df_12h['valuenum'].between(50, 100))]

    # Pivot to wide format
    vitals_matrix = vitals_df_12h.pivot_table(
            index='subject_id',
            columns='label',
            values='valuenum',
            aggfunc='mean',
            fill_value=np.nan).reset_index()

    all_vitals_matrix = vitals_matrix.round(2)
    #vitals_columns = all_vitals_matrix.columns.drop('subject_id')

    # # Aantal subject_ids waar Temperature een waarde heeft
    # num_temperature = vitals_matrix['Temperature'].notna().sum()
    # print(f"Aantal subject_ids met Temperature-waarde: {num_temperature}")

    # vitals_matrix.to_csv("vitals_check.csv")
    # Controleer uniekheid van subject_id
    # num_subjects = vitals_matrix['subject_id'].nunique()
    # total_rows = len(vitals_matrix)
    # print(f"Aantal unieke subject_ids: {num_subjects}")
    # print(f"Totaal aantal rijen in vitals_matrix: {total_rows}")

    return all_vitals_matrix

# if __name__ == "__main__":
#     vitals_matrix = get_vitals_matrix_12h()
#     vitals_matrix.to_csv("check_vitals2.csv")
#     print(vitals_matrix)
