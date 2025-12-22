import pandas as pd
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
PATH_DATA = REPO_ROOT / "data"
SEPSIS_CSV  = "sepsis_diagnose_time.csv"
CREAT_CSV   = "creatinine_over_time.csv"

def extract_creatinine(CREAT_CSV,SEPSIS_CSV):
    df_sepsis = pd.read_csv(PATH_DATA / SEPSIS_CSV)

    cols = ["subject_id", "charttime", "valuenum"]  # alleen relevante kolommen
    df_creat  = pd.read_csv(PATH_DATA / CREAT_CSV,usecols=cols)
    
    # 1. Lees sepsis data (je hoeft meestal niet te beperken, maar kan wel)
    subject_ids = set(df_sepsis["subject_id"])

    # 2. Filter alleen patiënten die ook in sepsis voorkomen
    filtered_df = df_creat[df_creat["subject_id"].isin(subject_ids)]
    
    #print("DEBUG: Rows after filtering by sepsis patients:", len(filtered_df))
    
    filtered_df = filtered_df[["subject_id","charttime","valuenum"]]
 
    return filtered_df

