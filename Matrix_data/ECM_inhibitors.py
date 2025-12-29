import pandas as pd
from pathlib import Path
import numpy as np

def get_ACE_ARB_matrix_12h():
    """
    Builds a 0/1 matrix of ACE and ARB drug administration within 12h before AKI onset.
    Rows: subject_id
    Columns: 'ACE', specific ACE drugs, 'ARB', specific ARB drugs
    Returns a pandas DataFrame.
    """

    # Paths
    REPO_ROOT = Path(__file__).resolve().parent.parent
    PATH_DATA = REPO_ROOT / "data"

    # Files
    drug_file = PATH_DATA / "ACE_ARB.csv"           
    aki_file = "AKI_subjects.csv"
    aki_onset_file = "AKI_stage_output.csv"

    # Read data
    drug_df = pd.read_csv(drug_file, dtype={'subject_id': str})
    aki_subject_df = pd.read_csv(aki_file, dtype={'subject_id': str})
    aki_onset_df = pd.read_csv(aki_onset_file, dtype={'subject_id': str})

    # Filter only AKI subjects
    aki_subjects = set(aki_subject_df['subject_id'])
    drug_df = drug_df[drug_df['subject_id'].isin(aki_subjects)]

    # Convert times to datetime
    drug_df['stoptime'] = pd.to_datetime(drug_df['stoptime'])
    aki_onset_df['AKI_time'] = pd.to_datetime(aki_onset_df['AKI_time'])

    # Merge AKI onset time
    drug_df = drug_df.merge(
        aki_onset_df[['subject_id','AKI_time']],
        on='subject_id',
        how='left')

    # Keep only drugs given within 12h before AKI onset
    drug_df['hours_before_AKI'] = (drug_df['AKI_time'] - drug_df['stoptime']).dt.total_seconds() / 3600
    drug_df_12h = drug_df[(drug_df['hours_before_AKI'] >= 0) & (drug_df['hours_before_AKI'] <= 12)].copy()

    # Define ACE and ARB drugs
    ACE_drugs = ['Captopril', 'Enalapril Maleate', 'Enalaprilat', 'Lisinopril', 'Quinapril']
    ARB_drugs = ['Losartan Potassium', 'Valsartan']

    # Add drug_class column
    def classify_drug(drug_name):
        if drug_name in ACE_drugs:
            return 'ACE'
        elif drug_name in ARB_drugs:
            return 'ARB'
        else:
            return 'Other'

    drug_df_12h['drug_class'] = drug_df_12h['drug'].apply(classify_drug)

    # Initialize matrix
    subject_ids = sorted(drug_df_12h['subject_id'].unique())
    columns_order = ['subject_id'] + ACE_drugs + ARB_drugs
    matrix = pd.DataFrame(0, index=range(len(subject_ids)), columns=columns_order)
    matrix['subject_id'] = subject_ids

    # Fill 1/0 for specific drugs
    for drug in ACE_drugs + ARB_drugs:
        matrix[drug] = matrix['subject_id'].isin(
            #drug_df_12h[drug_df_12h['drug'] == drug]['subject_id']).replace({True: 1, False: 0})  # Set NaN if drug wasn't given
            drug_df_12h[drug_df_12h['drug'] == drug]['subject_id']).eq(True).astype(int)  # Set NaN if drug wasn't given

    # Reorder columns
    matrix_ECM = matrix[columns_order]

    return matrix_ECM

# ACE_ARB=get_ACE_ARB_matrix_12h()
# print(ACE_ARB)
