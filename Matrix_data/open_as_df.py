import pandas as pd 

def open_as_df(csv_path, sepsis_path):
    """
    Load a CSV and keep only rows with subject_ids in the sepsis dataset.

    Parameters
    - csv_path : str or Path -> Path to the CSV to load.
    - sepsis_path : str or Path -> Path to the sepsis CSV with subject_ids.

    Returns
    - pandas.DataFrame -> Filtered DataFrame with only matching subject_ids.
    """

    df = pd.read_csv(csv_path)
    sepsis = pd.read_csv(sepsis_path)

    sepsis_id = sepsis['subject_id'].unique()

    filtered_df = df[df['subject_id'].isin(sepsis_id)]

    return filtered_df
