import sys
from pathlib import Path
import pandas as pd
from Tool_1.extract_data_pandas import extract_creatinine
from Tool_1.determining_baseline import peak_creat, compute_baseline
from Tool_1.determining_AKI import AKI_detection
from matrix_dataframe import build_feature_matrix
from Matrix_data.fill_final_matrix import fill_matrix_with_zeros
from code_dashboard.creatininevalues import build_creatinine_window_12h_aki
from code_dashboard.heart_rate import build_heartrate_12h_before_aki
from code_dashboard.medical_history import save_conditions_per_subject
from code_dashboard.medication import combine_all_to_one_csv_long_with_time

# Data that is used
REPO_ROOT   = Path(__file__).resolve().parent
PATH_DATA   = REPO_ROOT / "data"
SEPSIS_CSV  = "sepsis_diagnose_time.csv"
CREAT_CSV   = "creatinine_over_time.csv"
OUTPUT_CSV = "AKI_stage_output.csv"
OUTPUT_CSV2 = "AKI_subjects.csv"

def main():
    AKI_df = AKI_detection(SEPSIS_CSV,CREAT_CSV)
    AKI_df.to_csv(OUTPUT_CSV)
    print(f"Total amount of patients with sepsis: {AKI_df['subject_id'].nunique()}")
    AKI_subjects_df = (AKI_df.loc[AKI_df["AKI"], ["subject_id"]].drop_duplicates().assign(AKI_stage="AKI").reset_index(drop=True))
    AKI_subjects_df.to_csv(OUTPUT_CSV2)
    print(f"AKI output saved as: {OUTPUT_CSV}")
    print(f"AKI subjects saved as: {OUTPUT_CSV2}")
    matrix = build_feature_matrix()
    print(f"Amount of patients in matrix: {matrix['subject_id'].nunique()}")
    missing_values,medians_df,final_matrix = fill_matrix_with_zeros()
    #Print missing values greater than 0
    print("Missing values in some columns:")
    print(missing_values[missing_values['missing_values'] > 0])

    # Print which medians were used for filling
    print("Medians used for vitals:")
    print(medians_df)

    # The final matrix
    final_matrix.to_csv("matrix_filled.csv", index=False)

    # Making dashboard CSVs and save them in the folder csv_dashboard
    print("Generating creatinine CSV...")
    df_creatinine = build_creatinine_window_12h_aki()
    print("Generating heart rate CSV...")
    df_heartrate = build_heartrate_12h_before_aki()
    print("Generating medical history CSV...")
    df_conditions = save_conditions_per_subject()
    print("Generating medication CSVs...")
    df_med_binary, df_med_amounts = combine_all_to_one_csv_long_with_time()
    print("All dashboard CSVs generated successfully!")


# Zorg dat dit script alleen draait als het direct wordt uitgevoerd
if __name__ == "__main__":
    main()
