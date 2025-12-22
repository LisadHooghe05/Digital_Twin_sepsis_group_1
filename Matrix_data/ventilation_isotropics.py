import pandas as pd
import numpy as np
from pathlib import Path

from Tool_1.determining_AKI import AKI_detection


def matrix_dataframe_ventilationandisotropics():
    """Build patient feature matrix for inotropics and ventilation within 12h before AKI."""

    # =======================
    # PATHS & CONSTANTS
    # =======================
    REPO_ROOT = Path(__file__).resolve().parent.parent
    PATH_DATA = REPO_ROOT / "data"

    INOTROPICS = [
        'Dopamine', 'Norepinephrine', 'Epinephrine', 'Milrinone',
        'Metoprolol', 'Esmolol', 'Verapamil', 'Diltiazem'
    ]

    VENTILATIONS = [
        'Intubation', 'Non-invasive ventilation', 'Invasive ventilation'
    ]

    SEPSIS_CSV = "sepsis_diagnose_time.csv"
    CREATININE_CSV = "creatinine_over_time.csv"

    PROCEDUREEVENT_CSV = PATH_DATA / "procedureevents_sepsis.csv"
    INPUT_CSVS = [
        PATH_DATA / "inputevents_sepsis1.csv",
        PATH_DATA / "inputevents_sepsis2.csv",
        PATH_DATA / "inputevents_sepsis3.csv",
    ]

    # =======================
    # AKI (1x correct!)
    # =======================
    AKI_df, AKI_subjects_df = AKI_detection(SEPSIS_CSV, CREATININE_CSV)
    AKI_df["AKI_time"] = pd.to_datetime(AKI_df["AKI_time"], errors="coerce")

    aki_subjects = AKI_df.loc[AKI_df["AKI"], ["subject_id", "AKI_time"]]

    # =======================
    # INOTROPICS
    # =======================
    input_df = pd.concat(
        [pd.read_csv(f) for f in INPUT_CSVS],
        ignore_index=True
    )

    input_df.columns = input_df.columns.str.strip().str.lower()
    input_df["starttime"] = pd.to_datetime(input_df["starttime"], errors="coerce")
    input_df["item_label"] = input_df["item_label"].str.lower()

    df_in = (
        input_df
        .merge(aki_subjects, on="subject_id", how="inner")
    )

    df_in["hours_before_aki"] = (
        (df_in["AKI_time"] - df_in["starttime"])
        .dt.total_seconds() / 3600
    )

    df_in = df_in[
        (df_in["hours_before_aki"] >= 0) &
        (df_in["hours_before_aki"] <= 12)
    ]

    df_in = df_in[df_in["item_label"].isin([x.lower() for x in INOTROPICS])]

    inotropics_df = (
        df_in
        .pivot_table(
            index="subject_id",
            columns="item_label",
            values="amount",
            aggfunc="sum",
            fill_value=0
        )
        .reset_index()
    )

    # =======================
    # VENTILATION
    # =======================
    proc_df = pd.read_csv(PROCEDUREEVENT_CSV)
    proc_df.columns = proc_df.columns.str.strip().str.lower()

    proc_df["starttime"] = pd.to_datetime(proc_df["starttime"], errors="coerce")
    proc_df["endtime"] = pd.to_datetime(
        proc_df.get("endtime", proc_df.get("stoptime")),
        errors="coerce"
    )

    proc_df["item_label"] = proc_df["item_label"].str.lower().str.strip()

    df_v = proc_df.merge(aki_subjects, on="subject_id", how="inner")

    df_v["win_start"] = df_v["AKI_time"] - pd.Timedelta(hours=12)
    df_v["win_end"] = df_v["AKI_time"]

    df_v["clip_start"] = df_v[["starttime", "win_start"]].max(axis=1)
    df_v["clip_end"] = df_v[["endtime", "win_end"]].min(axis=1)

    df_v = df_v[df_v["clip_end"] > df_v["clip_start"]]

    df_v["minutes"] = (
        (df_v["clip_end"] - df_v["clip_start"])
        .dt.total_seconds() / 60
    )

    ventilation_df = pd.DataFrame(
        {"subject_id": aki_subjects["subject_id"].unique()}
    )

    for label in VENTILATIONS:
        lbl = label.lower()
        sub = df_v[df_v["item_label"] == lbl]

        if sub.empty:
            continue

        if lbl == "intubation":
            agg = (
                sub.groupby("subject_id")
                .size()
                .gt(0)
                .astype(int)
                .reset_index(name=label)
            )
        else:
            agg = (
                sub.groupby("subject_id")["minutes"]
                .sum()
                .reset_index(name=label)
            )

        ventilation_df = ventilation_df.merge(agg, on="subject_id", how="left")

    ventilation_df = ventilation_df.fillna(0)

    # =======================
    # FINAL MATRIX
    # =======================
    final_df = inotropics_df.merge(
        ventilation_df,
        on="subject_id",
        how="left"
    )

    return final_df
