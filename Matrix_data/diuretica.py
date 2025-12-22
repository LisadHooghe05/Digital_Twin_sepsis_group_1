from pathlib import Path
from Matrix_data.open_as_df import open_as_df
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_diuretics_matrix_12h():
    """
    Builds a wide-format matrix for diuretics administration within a specified
    number of hours before AKI onset.
    
    Parameters:
    - sepsis_data: DataFrame with all sepsis events
    - AKI_df: DataFrame with columns ['subject_id','AKI_stage']
    - AKI_onset_df: DataFrame with columns ['subject_id','AKI_onset_time']
    - diuretics_list: list of diuretics to keep (default common ones)
    - hours_before: time window in hours before AKI onset to include events
    
    Returns:
    - diuretics_matrix: pd.DataFrame in wide format (subject_id × diuretics)
    """
    #Load data
    REPO_ROOT = Path(__file__).resolve().parent.parent
    PATH_DATA = REPO_ROOT / "data"

    sepsis_path = PATH_DATA / "sepsis_diagnose_time.csv"
    input_paths = [
            PATH_DATA / "inputevents_sepsis1.csv",
            PATH_DATA / "inputevents_sepsis2.csv",
            PATH_DATA / "inputevents_sepsis3.csv"]
    sepsis_path = PATH_DATA / "sepsis_diagnose_time.csv"

    #Loading sepsis data
    sepsis_data = pd.concat([open_as_df(p, sepsis_path) for p in input_paths], ignore_index=True)

    def get_column_index(df, column_name):
        """
        Returns the position of a wanted column in the datafrome
        if the column does not exist, raises a ValueError
        """
        try: 
            return df.columns.get_loc(column_name)
        except KeyError:
            raise ValueError(f"Column'{column_name}' not found in Dataframe")

    df = sepsis_data 
    col_name = "item_label" # -> changes this to get another column for example item_category for fluids/intake
    col_index = get_column_index(df,col_name)
    #print(f"Column'{col_name}' is at index '{col_index}'")

    AKI_df = pd.read_csv("AKI_subjects.csv") # file with the AKI stages coupled to the subject_IDs
    AKI_onset_df=pd.read_csv("AKI_stage_output.csv")
    
    # Diuretics we want to keep from al the medicins
    diuretics = [
        'Bumetanide (Bumex)',
        'Furosemide (Lasix)',
        'Furosemide (Lasix) 250/50',
        'Torasemide','Metolazon','Chloortalidon']


    df_diu = sepsis_data[ sepsis_data['item_label'].isin(diuretics) ].copy()

    AKI_df['subject_id'] = AKI_df['subject_id'].astype(str)
    df_diu['subject_id'] = df_diu['subject_id'].astype(str)
    AKI_onset_df['subject_id'] = AKI_onset_df['subject_id'].astype(str)

    df_merged = df_diu.merge(
        AKI_df[['subject_id','AKI_stage']],
        on='subject_id',
        how='left'
    ).merge(
        AKI_onset_df[['subject_id','AKI_time']],
        on='subject_id',
        how='left')

    df_merged['endtime'] = pd.to_datetime(df_merged['endtime'])
    df_merged['AKI_time'] = pd.to_datetime(df_merged['AKI_time'])
    
    # Calculates amount of hours before AKI onset
    df_merged['hours_before_AKI'] = (df_merged['AKI_time'] - df_merged['endtime']).dt.total_seconds() / 3600

    # Filter: only doses withing 12 hours before AKI onset
    df_merged_12h = df_merged[(df_merged['hours_before_AKI'] >= 0) & (df_merged['hours_before_AKI'] <= 12)].copy()

    # Controle
    #print(df_merged_12h[['item_label','AKI_stage','amount','endtime','AKI_onset_time','hours_before_AKI']].head())

    # Pivot to wide format: rows = subject_id, columns = diuretica, value = sum of the amounts
    diuretics_matrix_12h = df_merged_12h.pivot_table(
        index='subject_id',
        columns='item_label',
        values='amount',
        aggfunc='sum').reset_index()

    diuretics_matrix_12h = diuretics_matrix_12h.round(2)

    return diuretics_matrix_12h

# diu = get_diuretics_matrix_12h()
# print(diu)
