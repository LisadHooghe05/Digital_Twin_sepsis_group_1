import pandas as pd
from pathlib import Path

def compute_ICU_stays_window():
    """
    Determine whether patients had an ICU stay within the 12 hours prior to AKI onset.

    Returns
    - pandas.DataFrame ->
        A dataframe with subject_id and ICU_stay_12hforakitime, where values
        indicate ICU stay duration (in days) within the 12-hour window before AKI.
        Patients without an ICU stay receive a value of 0.
    """

    # The Paths 
    REPO_ROOT = Path(__file__).resolve().parent.parent
    PATH_DATA = REPO_ROOT / "data"

    # Load AKI subjects 
    aki_subjects = pd.read_csv("AKI_subjects.csv")
    aki_subjects["subject_id"] = aki_subjects["subject_id"].astype(str)
    # print("Amount data AKI_stages_subject_AKI:", len(aki_subjects))

    # Load AKI stage output and the columns we want to use 
    aki_output = pd.read_csv("AKI_stage_output.csv", usecols=["subject_id", "AKI_time"])
    aki_output["subject_id"] = aki_output["subject_id"].astype(str)

    # Load ICU stays and select only the columns you need
    sepsis = pd.read_csv(PATH_DATA / "sepsis3_los_dod.csv", usecols=["subject_id", "stay_id", "intime", "outtime"])

    # Make from 'intime' and 'outtime' datetime
    sepsis["intime"] = pd.to_datetime(sepsis["intime"], errors="coerce")
    sepsis["outtime"] = pd.to_datetime(sepsis["outtime"], errors="coerce")

    # Save the subject_ids in both dataframes as a string
    aki_output["subject_id"] = aki_output["subject_id"].astype(str)
    sepsis["subject_id"] = sepsis["subject_id"].astype(str)

    # Merge the ICU stay data (intime, outtime, stay_id) with AKI output 
    merged = aki_output.merge(sepsis, on="subject_id", how="left")

    # Filter the merged DataFrame to keep only the subject_ids that are in aki_subjects
    merged = merged[merged["subject_id"].isin(aki_subjects["subject_id"])]

    # Remove the NaT rows in the merged DataFrame, rows where AKI_time or intime/outtime is missing
    merged = merged.dropna(subset=["AKI_time", "intime", "outtime"])
    # print("Amount of subject ids where AKI-time or intime/outtime is available:", len(merged))

    # Make sure that the AKI_time column will be saved as datetime
    merged["AKI_time"] = pd.to_datetime(merged["AKI_time"], errors="coerce")

    # Compute the start_point (12 hours before AKI_time)
    merged["start_point"] = merged["AKI_time"] - pd.Timedelta(hours=12)

    # Filter stays that overlap with the 12-hour window
    merged_window = merged[
        (merged["intime"] >= merged["start_point"]) &  # 'intime' needs to be inbetween start_point and AKI_time 
        (merged["intime"] <= merged["AKI_time"]) &    # 'intime' needs to be before or equal to AKI_time
        (merged["outtime"] >= merged["start_point"])  # 'outtime' needs to be after or equal to the starting_point
    ]
    merged_window = merged_window.copy()
    # Compute the amount of days between 'intime' and 'AKI_time' 
    merged_window["ICU_stay_12hforakitime"] = (merged_window["outtime"] - merged_window["intime"]).dt.days

    # Make a matrix with subject_id, AKI_time, intime, outtime and start_point
    matrix_ICU_stay_12hforakitime = merged_window[["subject_id", "stay_id", "intime", "outtime", "start_point", "AKI_time", "ICU_stay_12hforakitime"]]
    # print("Amount subject ids with ICU stay:",len(matrix_ICU_stay_12hforakitime))

    # Give subject_ids without ICU stay a value 0
    missing_subjects = set(aki_subjects["subject_id"]) - set(merged_window["subject_id"])

    # Create a DataFrame for subject_ids without an ICU stay and set ICU_stay_12h_for_aki_time to 0
    missing_rows = pd.DataFrame({
        "subject_id": list(missing_subjects),
        "ICU_stay_12hforakitime": 0
    })
    matrix_ICU_stay_12hforakitime = merged_window[["subject_id","ICU_stay_12hforakitime"]]

    # Add missing rows to the matrix
    matrix_ICU_stay_12hforakitime = pd.concat([matrix_ICU_stay_12hforakitime, missing_rows], ignore_index=True)
    # To save the matrix
    # matrix_ICU_stay_12hforakitime.to_csv("ICU_stay_matrix.csv", index=False)
    # print("Matrix opgeslagen als 'ICU_stay_matrix.csv'")
    # print("Amount patients with ICU_stay of 12 hours before AKI_time:", len(matrix_ICU_stay_12hforakitime))
   
    return matrix_ICU_stay_12hforakitime

# To run the code
# ICU_stay_matrix = compute_ICU_stays_window()
# print(ICU_stay_matrix)
