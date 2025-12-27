import sys
from pathlib import Path
import pandas as pd
from datetime import timedelta


# CONFIG (portable)
REPO_ROOT   = Path(__file__).resolve().parent.parent     
PATH_FUNCS  = REPO_ROOT                             
PATH_DATA   = REPO_ROOT / "data"                   
SEPSIS_CSV  = "sepsis_diagnose_time.csv"
CREAT_CSV   = "creatinine_over_time.csv"

# Ensure that your extract function can be imported
sys.path.append(str(PATH_FUNCS))
from Tool_1.extract_data_pandas import extract_creatinine

# Extract creatinine from other code
df = extract_creatinine(CREAT_CSV,SEPSIS_CSV)

def peak_creat(extraction_creatinine) -> pd.DataFrame:
    """
    Computes summary statistics of creatinine for sepsis patients within 7 days post-sepsis.

    Parameters
    - extraction_creatinine : pd.DataFrame
        DataFrame containing creatinine measurements with columns:
        'subject_id', 'charttime', 'valuenum'.

    Returns
    - summary : pd.DataFrame
        One row per patient with:
        - 'sepsis_time': first sepsis timestamp
        - 'n_creat_0to7d': number of creatinine measurements in 0–7 days post-sepsis
        - 'peak_creat_value_0to7d': maximum creatinine value in 0–7 days
        - 'peak_creat_time_0to7d': timestamp of peak creatinine
        - 'min_creat_before_48h': minimum creatinine in 48h before peak

    - sepsis_first : pd.DataFrame -> First sepsis timestamp per patient.
    - creat : pd.DataFrame -> Filtered creatinine measurements for patients present in sepsis_first.
    
    Notes
    - Only includes measurements within 7 days after sepsis.
    - Peak time is determined per patient, and 48-hour rolling minimum is calculated relative to the peak.
    """
    # Unique patients
    unique_patients = (
        extraction_creatinine["subject_id"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .to_frame(name="subject_id")
    )
    ids = pd.to_numeric(unique_patients["subject_id"], errors='coerce').astype("Int64")

    # Read sepsis CSV
    sepsis = pd.read_csv(PATH_DATA / SEPSIS_CSV, usecols=["subject_id", "suspected_infection_time"])
    sepsis["subject_id"] = pd.to_numeric(sepsis["subject_id"], errors='coerce').astype("Int64")

    # Convert sepsis times to datetime
    sepsis["suspected_infection_time"] = pd.to_datetime(
        sepsis["suspected_infection_time"], 
        errors="coerce", dayfirst=False  # set dayfirst=True if your data is DD/MM/YYYY
)
    sepsis_filtered = sepsis[sepsis["subject_id"].isin(ids)].copy()

    sepsis_first = (
        sepsis_filtered
        .sort_values(["subject_id", "suspected_infection_time"])
        .drop_duplicates(subset="subject_id", keep="first")
        .rename(columns={"suspected_infection_time": "sepsis_time"}))

    # Read creatinine CSV
    creat = pd.read_csv(PATH_DATA / CREAT_CSV, usecols=["subject_id", "charttime", "valuenum"] )
    creat["subject_id"] = pd.to_numeric(creat["subject_id"], errors='coerce').astype("Int64")

    # Convert charttime with AM/PM handling
    creat["charttime"] = pd.to_datetime(
        creat["charttime"], 
        errors="coerce", 
        dayfirst=False)
    
    # Keep only relevant patients
    creat = creat[creat["subject_id"].isin(sepsis_first["subject_id"])].copy()

    # Merge creatinine with sepsis time
    m = pd.merge(creat, sepsis_first[["subject_id", "sepsis_time"]], on='subject_id')

    # Ensure sepsis_time is datetime
    m["sepsis_time"] = pd.to_datetime(m["sepsis_time"], errors="coerce", dayfirst=False)

    # m.loc[m["charttime"].dt.year > 2100, "charttime"] -= pd.DateOffset(years=100)
    # m.loc[m["sepsis_time"].dt.year > 2100, "sepsis_time"] -= pd.DateOffset(years=100)

    # Filter 0–7 days after sepsis
    mask = (m["charttime"] >= m["sepsis_time"]) & (m["charttime"] <= m["sepsis_time"] + pd.Timedelta(days=7))
    
    creat_0to7d = m.loc[mask, ["subject_id", "sepsis_time", "charttime", "valuenum"]].copy()
    # creat_0to7d = creat_0to7d.rename(columns={"valuenum": "creatinine"})
    
    # Summary per subject
    summary = (
        creat_0to7d.groupby("subject_id", as_index=False)
                .agg(
                    sepsis_time=("sepsis_time", "first"),
                    n_creat_0to7d=("valuenum", "size"),
                    peak_creat_value_0to7d=("valuenum", "max",)))

    # Peak time per subject
    idx = creat_0to7d.groupby("subject_id")["valuenum"].idxmax()
    idx = idx.dropna().astype(int)  # verwijder NaN en maak indices geldig
    peak_time = creat_0to7d.loc[idx, ["subject_id", "charttime"]].rename(
    columns={"charttime": "peak_creat_time_0to7d"})

    # Ensure peak time is datetime
    peak_time["peak_creat_time_0to7d"] = pd.to_datetime(peak_time["peak_creat_time_0to7d"], errors="coerce")

    summary = summary.merge(peak_time, on="subject_id")

    # Map peak_time to creat_0to7d
    peak_time_lookup = summary.set_index("subject_id")["peak_creat_time_0to7d"]
    creat_0to7d = creat_0to7d.copy()
    creat_0to7d["peak_time"] = creat_0to7d["subject_id"].map(peak_time_lookup)
    creat_0to7d["peak_time"] = pd.to_datetime(creat_0to7d["peak_time"], errors="coerce")

    # 48-hour window mask
    mask_48h = (
        (creat_0to7d["charttime"] >= creat_0to7d["peak_time"] - pd.Timedelta(hours=48)) &
        (creat_0to7d["charttime"] <= creat_0to7d["peak_time"]))

    min48 = (creat_0to7d.loc[mask_48h]
             .groupby("subject_id", as_index=False)["valuenum"]
             .min()
             .rename(columns={"valuenum": "min_creat_before_48h"}))

    summary = summary.merge(min48, on="subject_id")

    return summary, sepsis_first, creat

summary, sepsis_first, creat = peak_creat(df)

def compute_baseline(sepsis_first, creat, summary) -> pd.DataFrame:
    """
    Compute the baseline serum creatinine for each patient relative to sepsis onset.

    The function determines a patient-specific baseline using multiple rules:
    1. Pre-sepsis (-7 to 0 days): if ≥2 creatinine measurements, take the lowest.
    2. Pre-sepsis longer windows (30, 90, 365 days): if no 7-day baseline, use median value.
    3. Post-sepsis (0–30 days, 0–365 days): only if pre-sepsis data insufficient, take earliest/minimum.

    Parameters
    - sepsis_first : pd.DataFrame -> DataFrame with columns ['subject_id', 'sepsis_time'] containing first sepsis time per patient.
    - creat : pd.DataFrame -> DataFrame with columns ['subject_id', 'charttime', 'creatinine'] with creatinine measurements.
    - summary : pd.DataFrame -> DataFrame containing peak creatinine metrics (from peak_creat) to be attached to baseline.

    Returns
    - pd.DataFrame
        DataFrame per patient with columns:
        - subject_id
        - sepsis_time
        - baseline_value : determined baseline creatinine
        - baseline_time  : time of baseline measurement
        - baseline_rule  : rule applied to determine baseline
        - peak_creat_value_0to7d : peak creatinine within 0–7 days post-sepsis
        - min_creat_before_48h    : minimum creatinine in 48h before peak
    """

    # Normalize to standard column names
    s = sepsis_first[["subject_id", "sepsis_time"]].copy()
    s["subject_id"] = pd.to_numeric(s["subject_id"], errors="coerce").astype("Int64")
    s["sepsis_time"] = pd.to_datetime(s["sepsis_time"], errors="coerce")

    c = creat.rename(columns={"valuenum": "creatinine"})[
        ["subject_id", "charttime", "creatinine"]
    ].copy()
    c["subject_id"] = pd.to_numeric(c["subject_id"], errors="coerce").astype("Int64")
    c["charttime"] = pd.to_datetime(c["charttime"], errors="coerce")

    # Filtering/sorting
    c = c.sort_values(["subject_id", "charttime"])
    
    groups = {sid: g for sid, g in c.groupby("subject_id")}

    rows = []
    for row in s.itertuples(index=False):
        sid = row.subject_id
        st  = row.sepsis_time
        g = groups.get(sid)

        baseline_value = None
        baseline_time  = None
        rule = "none"
        n_used = 0

        if g is not None and pd.notna(st):
            # PRE: -7..0 days, >=2 points -> lowest 
            win_7d = g[(g["charttime"] >= st - timedelta(days=7)) & (g["charttime"] <= st)]
            if len(win_7d) >= 2:
                idx_min = win_7d["creatinine"].idxmin()
                baseline_value = win_7d.loc[idx_min, "creatinine"]
                baseline_time  = win_7d.loc[idx_min, "charttime"]
                rule = "pre_7d_min"
                n_used = len(win_7d)
            else:
                # PRE: 30 / 90 / 365 days -> median 
                for days, rname in [(30, "pre_1m_median"), (90, "pre_3m_median"), (365, "pre_1y_median")]:
                    win = g[(g["charttime"] > st - timedelta(days=days)) & (g["charttime"] <= st)]
                    if not win.empty:
                        if win["creatinine"].notna().any():
                            med = float(win["creatinine"].median())
                            idx_med = (win["creatinine"] - med).abs().idxmin()

                            if pd.notna(idx_med):
                                baseline_value = med
                                baseline_time  = win.loc[idx_med, "charttime"]
                                rule = rname
                                n_used = len(win)
                                break


                # POST: only if PRE yielded no result 
                if rule == "none":
                    win_p1m = g[(g["charttime"] >= st) & (g["charttime"] <= st + timedelta(days=30))]
                    if not win_p1m.empty:
                        idx_min = win_p1m["creatinine"].idxmin()
                        baseline_value = win_p1m.loc[idx_min, "creatinine"]
                        baseline_time  = win_p1m.loc[idx_min, "charttime"]
                        rule = "post_1m_min"
                        n_used = len(win_p1m)
                    else:
                        win_p1y = g[(g["charttime"] > st) & (g["charttime"] <= st + timedelta(days=365))]
                        if not win_p1y.empty:
                            first_row = win_p1y.iloc[0]
                            baseline_value = first_row["creatinine"]
                            baseline_time  = first_row["charttime"]
                            rule = "post_1y_oldest"
                            n_used = len(win_p1y)

        rows.append({
            "subject_id": sid,
            "sepsis_time": st,
            "baseline_value": baseline_value,
            "baseline_time": baseline_time,
            "baseline_rule": rule,})
    baseline_info = pd.DataFrame(rows)
    new_summary = baseline_info.merge(
        summary[["subject_id", "peak_creat_value_0to7d", "min_creat_before_48h"]],
        on="subject_id",
        how="left")

    return new_summary

#Calculate baseline and attach peak value to baseline
baseline_df = compute_baseline(sepsis_first, creat, summary)

def main():
    # Creatinine by calling a function
    df = extract_creatinine(CREAT_CSV,SEPSIS_CSV)

    # Peak creatinine + summary
    summary, sepsis_first, creat = peak_creat(df)

    # Compute baseline 
    baseline_df = compute_baseline(sepsis_first, creat, summary)

    # Print or save
    #print(baseline_df.head())
    # baseline_df.to_csv(PATH_DATA / "baseline_summary.csv", index=False)

# Ensure this script runs only when executed directly
if __name__ == "__main__":
    main()
