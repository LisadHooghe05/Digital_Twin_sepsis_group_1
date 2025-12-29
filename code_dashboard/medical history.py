from pathlib import Path
import pandas as pd

def save_conditions_per_subject():
    """
    Build a long-format CSV of pre-conditions per sepsis patient with AKI.

    Each row corresponds to one condition of one patient (subject_id).
    """
    # Paths
    REPO_ROOT = Path(__file__).resolve().parent.parent
    PATH_DATA = REPO_ROOT / "data"

    conditions_path = PATH_DATA / "sepsis_preex_conditions.csv"
    aki_subjects_path = "AKI_subjects.csv"

    # Load AKI subjects
    aki_subjects = pd.read_csv(aki_subjects_path, dtype={'subject_id': str})

    # Load conditions
    conditions_df = pd.read_csv(conditions_path, dtype={'subject_id': str})

    # Keep only AKI sepsis patients
    conditions_df = conditions_df[conditions_df['subject_id'].isin(aki_subjects['subject_id'])]

    # Add indicator column if needed
    conditions_df['value'] = 1

    # Keep only necessary columns
    long_conditions = conditions_df[['subject_id', 'condition', 'value']]

    # Sort by subject_id
    long_conditions = long_conditions.sort_values(['subject_id', 'condition']).reset_index(drop=True)

    # Save to CSV
    output_path = REPO_ROOT / "csv_dashboard" / "conditions_per_subject.csv"
    output_path.parent.mkdir(exist_ok=True)
    long_conditions.to_csv(output_path, index=False, decimal=',')

    print(f"Long-format pre-conditions CSV saved to {output_path}")
    return long_conditions

# Example usage:
df_long = save_conditions_per_subject()
