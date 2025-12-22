import sys
from pathlib import Path
import pandas as pd
from Tool_1.extract_data_pandas import extract_creatinine
from Tool_1.determining_baseline import peak_creat, compute_baseline
from Tool_1.determining_AKI import AKI_detection
from Matrix_data.testmatrix import build_feature_matrix


# ==== CONFIG (portable) ====
REPO_ROOT   = Path(__file__).resolve().parent
PATH_DATA   = REPO_ROOT / "data"
SEPSIS_CSV  = "sepsis_diagnose_time.csv"
CREAT_CSV   = "creatinine_over_time.csv"
OUTPUT_CSV = "AKI_stage_output.csv"
OUTPUT_CSV2 ="AKI_subjects.csv"
# ===========================

def main():
    AKI_df,AKI_subjects_df = AKI_detection(SEPSIS_CSV,CREAT_CSV)
    AKI_df.to_csv(OUTPUT_CSV)
    AKI_subjects_df.to_csv(OUTPUT_CSV2)
    print(f"AKI output opgeslagen als: {OUTPUT_CSV}")
    matrix = build_feature_matrix()
    matrix.to_csv("matrix.csv")
    print(f"Amount subject_id's: {matrix['subject_id'].nunique()}")


# Zorg dat dit script alleen draait als het direct wordt uitgevoerd
if __name__ == "__main__":
    main()
