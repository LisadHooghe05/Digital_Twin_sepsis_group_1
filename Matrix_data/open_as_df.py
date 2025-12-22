import pandas as pd 

def open_as_df(csv_path, sepsis_path):
    df = pd.read_csv(csv_path)
    sepsis = pd.read_csv(sepsis_path)

    sepsis_id = sepsis['subject_id'].unique()

    filtered_df = df[df['subject_id'].isin(sepsis_id)]

    return filtered_df
