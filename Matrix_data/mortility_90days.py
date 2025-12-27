import pandas as pd
from pathlib import Path

def create_AKI_90day_matrix():
    """
    Returns a DataFrame with:
    - index: subject_id
    - column: died_within_90d_after_AKI (1 if died within 90 days, 0 otherwise)
    """

    # Paths
    REPO_ROOT = Path(__file__).resolve().parent.parent
    PATH_DATA = REPO_ROOT / "data"

    # Load AKI subjects
    aki_subjects = pd.read_csv("AKI_subjects.csv")
    aki_subjects["subject_id"] = aki_subjects["subject_id"].astype(str)
    # print("Amount data AKI_stages_subject_AKI:", len(aki_subjects))

    # # Load AKI stage output
    aki_output = pd.read_csv("AKI_stage_output.csv")
    aki_output["subject_id"] = aki_output["subject_id"].astype(str)
    # print(aki_output)

    # print("Amount data AKI_stage_output:", len(aki_output))

    # Amount of rows that have a AKI_time
    num_present = aki_output["AKI_time"].notna().sum()
    # print(num_present)

    # Load sepsis / dod data
    sepsis = pd.read_csv(PATH_DATA / "sepsis3_los_dod.csv")
    sepsis["subject_id"] = sepsis["subject_id"].astype(str)
    sepsis["dod"] = pd.to_datetime(sepsis["dod"], format="%Y-%m-%d", errors="coerce")
    # print("Amount data sepsis3_los_dod:", len(sepsis))

    # Filter only on the subject_ids out the csv file aki_subjects
    sepsis_filtered = sepsis[sepsis["subject_id"].isin(aki_subjects["subject_id"])]
    sepsis_filtered_unique = sepsis_filtered.drop_duplicates(subset="subject_id", keep="first")
    # print(sepsis_filtered_unique)
    # print("ammount aki_subjects",len(sepsis_filtered_unique)) 

    #sepsis_filtered_unique.to_csv("dodenaki.csv")
    aki_first = aki_output.sort_values("AKI_time").drop_duplicates(subset="subject_id")
    merged = pd.merge(sepsis_filtered_unique, aki_first[["subject_id", "AKI_time"]],on="subject_id", how="left")
    # print(merged)
    #merged.to_csv("merged.csv")
    amount = merged[merged['AKI_time'].notna()].shape[0]

    # print("ammount subject_id's with dod and AKI_time value",amount)

    # Keep only rows that have both death and AKI_time
    alive_subjects = merged.dropna(subset=["AKI_time"]).copy()
    # print("Number of remaining subject_ids:", len(alive_subjects))

    # Make from dod and AKI_time datetimes
    alive_subjects['dod'] = pd.to_datetime(alive_subjects['dod'], dayfirst=True, errors='coerce')
    alive_subjects['AKI_time'] = pd.to_datetime(alive_subjects['AKI_time'], dayfirst=False, errors='coerce')

    # Compute days until death
    alive_subjects["days_until_death"] = (alive_subjects["dod"] - alive_subjects["AKI_time"]).dt.days
    # alive_subjects.to_csv("daysuntildeath.csv")
    alive_subjects["days_until_death"] = alive_subjects["days_until_death"].clip(lower=0)

    alive_subjects["died_within_90d_after_AKI"] = alive_subjects["days_until_death"].between(0, 90).astype(int)
    #positive_values.to_csv("positive_values.csv")

    # Making the matrix with colums subject_id and died_within_90d_after_AKI
    matrix_mortality_90days = alive_subjects[["subject_id", "died_within_90d_after_AKI"]]
    # num_died_90d = alive_subjects["died_within_90d_after_AKI"].sum()
    # print("Amount of patients that dies within 90 days after AKI:, num_died_90d)
    # matrix_mortality_90days.to_csv("mortality_matrix90d.csv")
    # print(matrix_mortality_90days)
    # print("amount subjectids:",len(matrix_mortality_90days))

    return matrix_mortality_90days
