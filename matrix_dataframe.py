import pandas as pd
from pathlib import Path

def build_feature_matrix(repo_root=None):
    """
    Build the full patient feature matrix including AKI, vitals, medications, ventilation, 
    inotropics, presepsis conditions, mortality, and ICU stays.
    
    Parameters
    ----------
    repo_root : Path or str, optional
        Root folder of the repository. If None, defaults to the folder of this script.
    
    Returns
    -------
    pd.DataFrame
        Final patient matrix.
    """
    # the paths
    REPO_ROOT = Path(__file__).resolve().parent
    PATH_DATA = REPO_ROOT / "data"

    def add_df_to_matrix(matrix, df, key="subject_id", columns=None):
        """
        Quick additon of features by using index-join 
        """
        if key in df.columns:
            df = df.set_index(key)

        if columns is not None:
            df = df[columns]

        return matrix.join(df, how="left")


    # Read AKI subjects
    aki_subjects = (
        pd.read_csv("AKI_subjects.csv", dtype={"subject_id": str})
        [["subject_id"]]
        .drop_duplicates()
        .set_index("subject_id")
    )

    print(f"Amount of sepsis patients with AKI: {len(aki_subjects)}")


    # Filter subjects on vitals
    from Matrix_data.combined_vitals import get_vitals_matrix_12h
    vitals_matrix = get_vitals_matrix_12h()
    vitals_matrix["subject_id"] = vitals_matrix["subject_id"].astype(str)
    valid_subjects = vitals_matrix["subject_id"].unique()
    aki_subjects = aki_subjects.loc[
        aki_subjects.index.isin(valid_subjects)]
    print(f"Amount of patients after filtering on vitals: {len(aki_subjects)}")

    # Init matrix
    matrix = aki_subjects.copy()

    # Adding gender to init matrix
    gender_df = pd.read_csv(PATH_DATA / "gender.csv", dtype={"subject_id": str})
    gender_df["gender"] = gender_df["gender"].map({"M": 0, "F": 1}).astype("Int8")
    matrix = add_df_to_matrix(matrix, gender_df)

    # Adding age
    from Matrix_data.age import get_age_12h_before_AKI
    age_df = get_age_12h_before_AKI()
    matrix = add_df_to_matrix(matrix, age_df)

    # Add vitals
    matrix = add_df_to_matrix(matrix, vitals_matrix)

    # Add fluids
    from Matrix_data.fluid import get_fluid_matrix_12h
    fluid_df = get_fluid_matrix_12h()
    matrix = add_df_to_matrix(matrix, fluid_df)

    # Add diuretics
    from Matrix_data.diuretica import get_diuretics_matrix_12h
    diuretics_df = get_diuretics_matrix_12h()
    matrix = add_df_to_matrix(matrix, diuretics_df)

    from Matrix_data.diuretic2 import get_diuretics2_matrix_12h
    diuretics2_df = get_diuretics2_matrix_12h()
    matrix = add_df_to_matrix(matrix, diuretics2_df)

    # Add ACE / ARB
    from Matrix_data.ECM_inhibitors import get_ACE_ARB_matrix_12h
    ace_arb_df = get_ACE_ARB_matrix_12h()
    matrix = add_df_to_matrix(matrix, ace_arb_df)

    # Add vasopressors
    from Matrix_data.vasopressors import get_vasopressor_matrix_12h
    vasopressor_df = get_vasopressor_matrix_12h()
    matrix = add_df_to_matrix(matrix, vasopressor_df)

    # Add antibiotics
    from Matrix_data.antibiotica_dataframe import antibiotica_df, other_meds_df

    matrix = add_df_to_matrix(matrix, antibiotica_df())
    matrix = add_df_to_matrix(matrix, other_meds_df())


    # Add ventilation and isotropics
    from Matrix_data.ventilation_isotropics import build_patient_feature_matrix
    matrix=add_df_to_matrix(matrix, build_patient_feature_matrix())

    # Add presepsis conditions of patients
    from Matrix_data.presepsis_conditions import get_conditions_matrix_sepsis
    conditions_df = get_conditions_matrix_sepsis()
    matrix = add_df_to_matrix(matrix, conditions_df)

    # Add mortility
    from Matrix_data.mortility_90days import create_AKI_90day_matrix
    matrix = add_df_to_matrix(matrix, create_AKI_90day_matrix())


    # Add ICU stay within 12 hours before AKI_time and AKI_time
    from Matrix_data.ICU_stay_12hforakitime import compute_ICU_stays_window
    matrix = add_df_to_matrix(matrix, compute_ICU_stays_window())

    # Make a nice matrix that is usable
    matrix.reset_index(inplace=True)

    # Save matrix
    #matrix.to_csv("matrix_overview.csv", index=False)
    #print(f"Amount subject_id's: {matrix['subject_id'].nunique()}")

    exclude = {"subject_id", "gender", "age_12h_before_AKI"}
    cols_to_check = [c for c in matrix.columns if c not in exclude]

    mask_empty = (matrix[cols_to_check].isna() | (matrix[cols_to_check] == 0)).all(axis=1)
    matrix = matrix.loc[~mask_empty]

    matrix.to_csv("matrix_overview_without_all_NaN.csv", index=False)

    # print(
    #     "Amount rows where all columns (besides subject_id, gender, age) are NaN:",
    #     mask_empty.sum()
    # )

    #print(f"Amount of patients after filtering out the patients with only NaN: {matrix['subject_id'].nunique()}")

    return matrix



