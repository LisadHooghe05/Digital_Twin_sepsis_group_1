from pathlib import Path
import pandas as pd

def get_conditions_matrix_sepsis():
    """
    Build a wide-format matrix of conditions for sepsis patients with AKI.

    Rows are patients (subject_id), columns are conditions, 
    and values are 1 if the condition is present, 0 otherwise.

    Returns
    - pandas.DataFrame -> Conditions matrix for sepsis patients with AKI.
    """
    # The Paths
    REPO_ROOT = Path(__file__).resolve().parent.parent
    PATH_DATA = REPO_ROOT / "data"

    conditions_path = PATH_DATA / "sepsis_preex_conditions.csv"
    sepsis_path = PATH_DATA / "sepsis_diagnose_time.csv"
    
    # Load AKI subjects
    aki_subjects = pd.read_csv("AKI_subjects.csv")
    aki_subjects["subject_id"] = aki_subjects["subject_id"].astype(str)
    # print("Amount data AKI_stages_subject_AKI:", len(aki_subjects))

    # Load conditions
    conditions_df = pd.read_csv(conditions_path)
    conditions_df['subject_id'] = conditions_df['subject_id'].astype(str)

    # Keep only sepsis AKI patients
    conditions_df = conditions_df.merge(
        aki_subjects,
        on='subject_id',
        how='inner')

    # Add indicator column (1 if condition present)
    conditions_df['value'] = 1

    # Pivot to wide format (conditions as columns)
    conditions_matrix = conditions_df.pivot_table(
        index='subject_id',
        columns='condition',
        values='value',
        aggfunc='max').reset_index()

    # Merge with all AKI subjects to ensure all are included, even when they do not have a condition
    conditions_matrix = pd.merge(aki_subjects[['subject_id']], conditions_matrix, 
                                 on='subject_id', how='left')
    # # Fill NaN values with 0
    # conditions_matrix = conditions_matrix.fillna(0)

    # Convert non-NaN values to 1, NaN to 0
    conditions_matrix = conditions_matrix.eq(conditions_matrix).astype(int) 

    # Save to CSV
    # conditions_matrix.to_csv("conditions_matrix.csv", index=False)

    # Ensure that 'subject_id' in conditions_matrix has the same data type
    conditions_matrix['subject_id'] = conditions_matrix['subject_id'].astype(str)

    return conditions_matrix

# # Run the code
# conditions = get_conditions_matrix_sepsis()
# print(conditions)
