import pandas as pd
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
PATH_DATA = REPO_ROOT / "data"
SEPSIS_CSV  = "sepsis_diagnose_time.csv"
CREAT_CSV   = "creatinine_over_time.csv"

def extract_creatinine(CREAT_CSV,SEPSIS_CSV):
    """
    Extracts creatinine measurements for patients present in a sepsis dataset.

    Parameters
    - CREAT_CSV : str
        Filename of the CSV containing creatinine measurements. Expected columns: 
        'subject_id', 'charttime', 'valuenum'.
    - SEPSIS_CSV : str -> Filename of the CSV containing sepsis patient data. Expected column: 'subject_id'.

    Returns
    - pd.DataFrame
        Filtered creatinine DataFrame containing only patients present in the sepsis dataset.
        Columns: 'subject_id', 'charttime', 'valuenum'.
    """

    df_sepsis = pd.read_csv(PATH_DATA / SEPSIS_CSV)

    cols = ["subject_id", "charttime", "valuenum"]  # alleen relevante kolommen
    df_creat  = pd.read_csv(PATH_DATA / CREAT_CSV,usecols=cols)
    
    # Read sepsis data
    subject_ids = set(df_sepsis["subject_id"])

    # Filter only patients who also appear in sepsis
    filtered_df = df_creat[df_creat["subject_id"].isin(subject_ids)]
    
    #print("DEBUG: Rows after filtering by sepsis patients:", len(filtered_df))
    
    filtered_df = filtered_df[["subject_id","charttime","valuenum"]]
 
    return filtered_df

