# Determining AKI based on creatinine baseline
import pandas as pd
from pathlib import Path
import sys

REPO_ROOT_AKI   = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT_AKI))

from Tool_1.determining_baseline import compute_baseline
from Tool_1.determining_baseline import peak_creat
from Tool_1.extract_data_pandas import extract_creatinine

PATH_DATA   = REPO_ROOT_AKI / "data"                    
SEPSIS_CSV  = "sepsis_diagnose_time.csv"
CREAT_CSV   = "creatinine_over_time.csv"

def AKI_detection(SEPSIS_CSV,CREAT_CSV) -> pd.DataFrame:
    """
    Detect acute kidney injury (AKI) in sepsis patients using creatinine measurements.

    AKI is identified using two criteria relative to sepsis onset:
    1. **48-hour criterion**: Increase in serum creatinine ≥ 0.3 mg/dL within any 48-hour window after sepsis.
    2. **7-day criterion**: Serum creatinine ≥ 1.5 × baseline within 7 days after sepsis.

    The function returns a patient-level DataFrame with AKI indicators, onset times, 
    and days from sepsis to AKI.

    Parameters
    - SEPSIS_CSV : str -> Path to the CSV file containing sepsis diagnosis times.
    - CREAT_CSV : str -> Path to the CSV file containing creatinine measurements (columns: subject_id, charttime, valuenum).

    Returns
    pd.DataFrame
        A DataFrame indexed by patient (`subject_id`) containing:
        - baseline_value: Baseline creatinine for each patient
        - AKI_48h, AKI_7d: Boolean flags for AKI per criterion
        - AKI_time_48h, AKI_time_7d: Onset time per criterion
        - AKI: Boolean flag if AKI detected by any criterion
        - AKI_time: Earliest AKI onset time
        - days_to_AKI: Days from sepsis onset to AKI
        - AKI_status: 'AKI' or 'No AKI'
    """

    # Upstream pipeline 
    df = extract_creatinine(CREAT_CSV,SEPSIS_CSV)
    summary, sepsis_first, creat = peak_creat(df)
    AKI_df = compute_baseline(sepsis_first, creat, summary).copy()

    # Prepare creatinine + sepsis_time (1x merge, not 2 times) 
    creat2 = creat.copy()

    # Expected columns from your creatinine data: subject_id, charttime, valuenum (-> creatinine)
    creat2["valuenum"] = pd.to_numeric(creat2["valuenum"], errors="coerce")
    creat2["charttime"] = pd.to_datetime(creat2["charttime"], errors="coerce")

    sepsis_first2 = sepsis_first.copy()
    sepsis_first2["sepsis_time"] = pd.to_datetime(sepsis_first2["sepsis_time"], errors="coerce")

    creat2 = creat2.merge(
        sepsis_first2[["subject_id", "sepsis_time"]],
        on="subject_id",
        how="left")

    # Drop rows that cannot be used
    creat2 = creat2.dropna(subset=["subject_id", "charttime", "sepsis_time", "valuenum"])

    # Only post-sepsis
    creat_post = creat2.loc[creat2["charttime"] >= creat2["sepsis_time"]].copy()
    creat_post = creat_post.sort_values(["subject_id", "charttime"]).reset_index(drop=True)

    # 48h criterium: SCr(t) - min(SCr in prior 48h) >= 0.3 
    if creat_post.empty:
        # Create empty output columns so that merges do not fail        AKI_df["AKI_48h"] = False
        AKI_df["AKI_time_48h"] = pd.NaT
    else:
        # Rolling minimum per patient; to_numpy() prevents reindexing issues during assignment
        roll_min_48h = (
            creat_post
            .groupby("subject_id", sort=False)
            .rolling("48h", on="charttime")["valuenum"]
            .min()
            .reset_index(level=0, drop=True))

        creat_post["min_48h"] = roll_min_48h.to_numpy()
        creat_post["AKI_hit_48h"] = (creat_post["valuenum"] - creat_post["min_48h"]) >= 0.3

        aki48_any = (
            creat_post.groupby("subject_id")["AKI_hit_48h"]
            .any()
            .reset_index(name="AKI_48h"))
        aki48_time = (
            creat_post.loc[creat_post["AKI_hit_48h"]]
            .groupby("subject_id")["charttime"]
            .min()
            .reset_index(name="AKI_time_48h"))

        AKI_df = AKI_df.merge(aki48_any, on="subject_id", how="left")
        AKI_df = AKI_df.merge(aki48_time, on="subject_id", how="left")
        # AKI_df["AKI_48h"] = AKI_df["AKI_48h"].fillna(False)
        AKI_df["AKI_48h"] = AKI_df["AKI_48h"].eq(True)



    # 7-day criterion: >= 1.5 * baseline within 7 days after sepsis 
    # Merge only baseline_value (DO NOT merge sepsis_time again)
    creat_post = creat_post.merge(
        AKI_df[["subject_id", "baseline_value"]],
        on="subject_id",
        how="left")

    # Restrict to 0-7 days after sepsis for the 7-day criterion
    creat_0to7d = creat_post.loc[
        creat_post["charttime"] <= (creat_post["sepsis_time"] + pd.Timedelta(days=7))].copy()

    if creat_0to7d.empty:
        AKI_df["AKI_7d"] = False
        AKI_df["AKI_time_7d"] = pd.NaT
    else:
        creat_0to7d["AKI_hit_7d"] = (
            creat_0to7d["baseline_value"].notna()
            & (creat_0to7d["valuenum"] >= 1.5 * creat_0to7d["baseline_value"]))

        aki7_any = (
            creat_0to7d.groupby("subject_id")["AKI_hit_7d"]
            .any()
            .reset_index(name="AKI_7d"))
        aki7_time = (
            creat_0to7d.loc[creat_0to7d["AKI_hit_7d"]]
            .groupby("subject_id")["charttime"]
            .min()
            .reset_index(name="AKI_time_7d"))

        AKI_df = AKI_df.merge(aki7_any, on="subject_id", how="left")
        AKI_df = AKI_df.merge(aki7_time, on="subject_id", how="left")
        # AKI_df["AKI_7d"] = AKI_df["AKI_7d"].fillna(False)
        AKI_df["AKI_7d"] = AKI_df["AKI_7d"].eq(True)


    # Combine binary AKI + onset time 
    if "AKI_48h" not in AKI_df.columns:
        AKI_df["AKI_48h"] = False
        AKI_df["AKI_time_48h"] = pd.NaT
    if "AKI_7d" not in AKI_df.columns:
        AKI_df["AKI_7d"] = False
        AKI_df["AKI_time_7d"] = pd.NaT

    AKI_df["AKI"] = AKI_df["AKI_48h"] | AKI_df["AKI_7d"]
    AKI_df["AKI_time"] = AKI_df[["AKI_time_48h", "AKI_time_7d"]].min(axis=1)

    AKI_df["days_to_AKI"] = (
        (AKI_df["AKI_time"] - AKI_df["sepsis_time"]).dt.total_seconds() / (3600 * 24))
    AKI_df["AKI_status"] = AKI_df["AKI"].map(
    {True: "AKI", False: "No AKI"})

    return AKI_df



def quick_check_aki_only(
    original_path="AKI_stage_output.csv",
    new_path="AKI_binary_output_check_1.csv",
    time_col="AKI_time",
    aki_filter="both_true",   # "orig_true" | "new_true" | "both_true" | "either_true"
    tolerance="0min"):
    df_o = pd.read_csv(original_path)
    df_n = pd.read_csv(new_path)

    df_o.columns = df_o.columns.str.strip()
    df_n.columns = df_n.columns.str.strip()

    # ORIG: Determine AKI from AKI_stage
    if "AKI_stage" not in df_o.columns:
        raise KeyError(f"ORIG mist 'AKI_stage'. Beschikbaar: {df_o.columns.tolist()}")

    stage = df_o["AKI_stage"].astype(str).str.strip().str.lower()
    df_o["AKI"] = ~stage.isin(["no aki", "aki determination not possible", "nan", "none"])

    if time_col not in df_o.columns:
        raise KeyError(f"ORIG mist '{time_col}'. Beschikbaar: {df_o.columns.tolist()}")

    # Parse time (strip empty strings)
    df_o[time_col] = pd.to_datetime(
        df_o[time_col].astype(str).str.strip().replace({"": None, "NaT": None, "nan": None}),
        errors="coerce")

    # NEW: AKI kolom 
    if "AKI" in df_n.columns:
        df_n["AKI"] = df_n["AKI"].astype(str).str.lower().isin(["true", "1", "yes", "aki"])
    elif "AKI_status" in df_n.columns:
        df_n["AKI"] = df_n["AKI_status"].astype(str).str.lower().eq("aki")
    else:
        raise KeyError(f"NEW mist 'AKI' (of 'AKI_status'). Beschikbaar: {df_n.columns.tolist()}")

    if time_col not in df_n.columns:
        raise KeyError(f"NEW mist '{time_col}'. Beschikbaar: {df_n.columns.tolist()}")

    df_n[time_col] = pd.to_datetime(
        df_n[time_col].astype(str).str.strip().replace({"": None, "NaT": None, "nan": None}),
        errors="coerce")

    # EXTRA PRINT: Original missing values overall (AKI-only)
    orig_aki = df_o[df_o["AKI"]]
    new_aki = df_n[df_n["AKI"]]
    print(f"ORIG AKI totaal: {len(orig_aki)}")
    print(f"NEW AKI totaal: {len(new_aki)}")
    print(f"ORIG AKI_time missing (alle ORIG AKI): {int(orig_aki[time_col].isna().sum())}")

    # Merge for comparison
    df_o = df_o[["subject_id", "AKI", time_col]].copy()
    df_n = df_n[["subject_id", "AKI", time_col]].copy()

    m = df_o.merge(df_n, on="subject_id", how="inner", suffixes=("_orig", "_new"))

    orig_aki_col = "AKI_orig"
    new_aki_col  = "AKI_new"
    orig_t = f"{time_col}_orig"
    new_t  = f"{time_col}_new"

    # Select filter
    if aki_filter == "orig_true":
        m = m[m[orig_aki_col]]
    elif aki_filter == "new_true":
        m = m[m[new_aki_col]]
    elif aki_filter == "both_true":
        m = m[m[orig_aki_col] & m[new_aki_col]]
    elif aki_filter == "either_true":
        m = m[m[orig_aki_col] | m[new_aki_col]]
    else:
        raise ValueError("aki_filter moet zijn: orig_true | new_true | both_true | either_true")

    tol = pd.Timedelta(tolerance)
    m["time_diff"] = (m[new_t] - m[orig_t]).abs()
    m["time_match"] = m["time_diff"].le(tol)

    n_total = len(m)
    n_match = int(m["time_match"].sum())
    n_missing_new = int(m[new_t].isna().sum())
    n_missing_orig = int(m[orig_t].isna().sum())
    n_mismatch = int((~m["time_match"] & m[new_t].notna() & m[orig_t].notna()).sum())

    # Your old output
    print(f"\nAKI filter: {aki_filter}")
    print(f"Totaal in vergelijking:                         {n_total}")
    print(f"Match binnen tolerance ({tolerance}):           {n_match}")
    print(f"NEW mist {time_col} (NaT):                      {n_missing_new}")
    print(f"ORIG mist {time_col} (NaT) binnen filter:       {n_missing_orig}")
    print(f"Mismatch (beide hebben tijd maar anders):       {n_mismatch}")

    return m.sort_values(["time_match", "time_diff"], ascending=[True, False])

