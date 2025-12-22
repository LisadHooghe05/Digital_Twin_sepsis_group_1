import pandas as pd
from pathlib import Path

def compute_ICU_stays_window():
    """
    Voor elke AKI subject_id:
    - Voeg de 'intime' en 'outtime' kolommen van sepsis toe aan aki_output
    - Bereken het aantal dagen tussen 'intime' en 'AKI_time' als 'intime' binnen 12 uur vóór AKI_time valt
    - Maak een matrix met 'subject_id' en het aantal dagen
    """

    # Zet de paden voor je data
    REPO_ROOT = Path(__file__).resolve().parent.parent
    PATH_DATA = REPO_ROOT / "data"

    # --- Load AKI subjects ---
    aki_subjects = pd.read_csv("AKI_subjects.csv")
    aki_subjects["subject_id"] = aki_subjects["subject_id"].astype(str)
    # print("Amount data AKI_stages_subject_AKI:", len(aki_subjects))

    # --- Load AKI stage output (en selecteer alleen de benodigde kolommen) ---
    aki_output = pd.read_csv("AKI_stage_output.csv", usecols=["subject_id", "AKI_time"])
    aki_output["subject_id"] = aki_output["subject_id"].astype(str)

    # --- Load ICU stays (sepsis data) and select only the columns you need ---
    sepsis = pd.read_csv(PATH_DATA / "sepsis3_los_dod.csv", usecols=["subject_id", "stay_id", "intime", "outtime"])

    # Zet 'intime' en 'outtime' om naar datetime
    sepsis["intime"] = pd.to_datetime(sepsis["intime"], errors="coerce")
    sepsis["outtime"] = pd.to_datetime(sepsis["outtime"], errors="coerce")

    # Zorg ervoor dat subject_id in beide DataFrames als string wordt opgeslagen
    aki_output["subject_id"] = aki_output["subject_id"].astype(str)
    sepsis["subject_id"] = sepsis["subject_id"].astype(str)

    # --- Merge de ICU stay gegevens (intime, outtime, stay_id) met AKI output ---
    merged = aki_output.merge(sepsis, on="subject_id", how="left")

    # --- Filter de merged DataFrame om alleen de subject_id's die ook in aki_subjects voorkomen ---
    merged = merged[merged["subject_id"].isin(aki_subjects["subject_id"])]

    # Verwijder NaT rijen in de merged DataFrame (rijen waar AKI_time of intime/outtime ontbreekt)
    merged = merged.dropna(subset=["AKI_time", "intime", "outtime"])
    # print("Amount of subject ids where AKI-time or intime/outtime is available:", len(merged))

    # --- Zorg ervoor dat de AKI_time kolom als datetime wordt opgeslagen ---
    merged["AKI_time"] = pd.to_datetime(merged["AKI_time"], errors="coerce")

    # --- Bereken het startpunt (12 uur vóór AKI_time) ---
    merged["start_point"] = merged["AKI_time"] - pd.Timedelta(hours=12)

    # --- Filter de stays die overlap hebben met het 12-uurs venster ---
    merged_window = merged[
        (merged["intime"] >= merged["start_point"]) &  # 'intime' moet tussen start_point en AKI_time liggen
        (merged["intime"] <= merged["AKI_time"]) &    # 'intime' moet voor of gelijk aan AKI_time liggen
        (merged["outtime"] >= merged["start_point"])  # 'outtime' moet na of gelijk aan start_point liggen
    ]

    # --- Bereken het aantal dagen tussen 'intime' en 'AKI_time' ---
    merged_window["ICU_stay_12hforakitime"] = (merged_window["outtime"] - merged_window["intime"]).dt.days

    # --- Maak de matrix met subject_id, AKI_time, intime, outtime en start_point ---
    matrix_ICU_stay_12hforakitime = merged_window[["subject_id", "stay_id", "intime", "outtime", "start_point", "AKI_time", "ICU_stay_12hforakitime"]]
    # print("Amount subject ids with ICU stay:",len(matrix_ICU_stay_12hforakitime))

        # --- Voeg de subject_id's zonder ICU stay toe met een waarde van 0 ---
    missing_subjects = set(aki_subjects["subject_id"]) - set(merged_window["subject_id"])

    # Maak een DataFrame van de subject_id's zonder ICU stay en voeg een waarde van 0 toe voor ICU_stay_12hforakitime
    missing_rows = pd.DataFrame({
        "subject_id": list(missing_subjects),
        "ICU_stay_12hforakitime": 0
    })
    matrix_ICU_stay_12hforakitime = merged_window[["subject_id","ICU_stay_12hforakitime"]]
    # Voeg de nieuwe rijen toe aan de matrix
    matrix_ICU_stay_12hforakitime = pd.concat([matrix_ICU_stay_12hforakitime, missing_rows], ignore_index=True)
    # --- Opslaan ---
    # matrix_ICU_stay_12hforakitime.to_csv("ICU_stay_matrix.csv", index=False)
    # print("Matrix opgeslagen als 'ICU_stay_matrix.csv'")
    # print("Amount patients with ICU_stay of 12 hours before AKI_time:", len(matrix_ICU_stay_12hforakitime))
   
    return matrix_ICU_stay_12hforakitime

# # --- Run ---
# ICU_stay_matrix = compute_ICU_stays_window()
# print(ICU_stay_matrix)
