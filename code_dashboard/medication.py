import pandas as pd
from pathlib import Path

def load_aki_info(REPO_ROOT):
    aki_subjects = pd.read_csv("AKI_subjects.csv", dtype={'subject_id': str})
    aki_times = pd.read_csv("AKI_stage_output.csv", dtype={'subject_id': str}, parse_dates=["AKI_time"])
    aki_times['window_start'] = aki_times['AKI_time'] - pd.Timedelta(hours=12)
    return aki_subjects, aki_times

def get_subject_ids_from_vitals(PATH_DATA, aki_subjects):
    vitals_files = [PATH_DATA / f"vitals{i}.csv" for i in [1,2,3]]
    sepsis_df = pd.read_csv(PATH_DATA / "sepsis_diagnose_time.csv", dtype={'stay_id': str, 'subject_id': str})
    vitals_df = pd.concat([pd.read_csv(f, dtype={'stay_id': str}) for f in vitals_files], ignore_index=True)
    vitals_df = vitals_df.merge(sepsis_df[['stay_id', 'subject_id']], on='stay_id', how='left')
    vitals_df = vitals_df[vitals_df['subject_id'].isin(set(aki_subjects['subject_id']))]
    return vitals_df['subject_id'].unique()

def combine_all_to_one_csv_long_with_time():
    REPO_ROOT = Path(__file__).resolve().parent.parent
    PATH_DATA = REPO_ROOT / "data"

    # print("Loading AKI info...")
    aki_subjects, aki_times = load_aki_info(REPO_ROOT)
    # print(f"Found {len(aki_subjects)} AKI subjects")

    # print("Getting subject IDs from vitals...")
    subject_ids = get_subject_ids_from_vitals(PATH_DATA, aki_subjects)
    # print(f"{len(subject_ids)} subjects found in vitals")

    def process_medications(df, med_column, amount_column=None):
        if df.empty:
            return pd.DataFrame(columns=['subject_id', 'medication', 'amount', 'starttime'])
        df = df.copy()
        df['starttime'] = pd.to_datetime(df.get('starttime', pd.NaT), errors='coerce')
        df['endtime'] = pd.to_datetime(df.get('endtime', pd.NaT), errors='coerce')
        if amount_column and amount_column in df.columns:
            df['amount'] = pd.to_numeric(df[amount_column], errors='coerce')
        else:
            df['amount'] = None  # binaire medicatie
        df = df.rename(columns={med_column: 'medication'})
        return df[['subject_id', 'medication', 'amount', 'starttime']]

    # --- Medications ---

    # Antibiotics
    # print("Processing antibiotics...")
    ab_paths = [PATH_DATA / f"inputevents_sepsis{i}.csv" for i in [1,2,3]]
    antibiotics = [
        "Vancomycin", "Teicoplanin", "Co-trimoxazole", "Sulfadiazine",
        "Sulfacetamide", "Gentamicin", "Tobramycin", "Amikacin",
        "Streptomycin", "Piperacillin", "Meropenem", "Cefepime",
        "Flucloxacillin", "Methicillin", "Bactrim (SMX/TMP)"
    ]
    ab_dfs = []
    for path in ab_paths:
        df = pd.read_csv(path, dtype={'subject_id': str})
        df = df[df['item_label'].isin(antibiotics)]
        ab_dfs.append(process_medications(df, 'item_label', 'value'))
    df_ab_long = pd.concat(ab_dfs, ignore_index=True)

    # Diuretics
    # print("Processing diuretics...")
    diu_paths = [PATH_DATA / f"inputevents_sepsis{i}.csv" for i in [1,2,3]]
    diuretics = [
        'Bumetanide (Bumex)', 'Furosemide (Lasix)', 'Furosemide (Lasix) 250/50',
        'Torasemide', 'Metolazon', 'Chloortalidon'
    ]
    diu_dfs = []
    for path in diu_paths:
        df = pd.read_csv(path, dtype={'subject_id': str})
        df = df[df['item_label'].isin(diuretics)]
        diu_dfs.append(process_medications(df, 'item_label', 'amount'))
    df_diu_long = pd.concat(diu_dfs, ignore_index=True)

    # ACE/ARB
    # print("Processing ACE/ARB...")
    ace_arb_df = pd.read_csv(PATH_DATA / "ACE_ARB.csv", dtype={'subject_id': str})
    ace_drugs = ['Captopril', 'Enalapril Maleate', 'Enalaprilat', 'Lisinopril', 'Quinapril']
    arb_drugs = ['Losartan Potassium', 'Valsartan']
    ace_arb_df = ace_arb_df[ace_arb_df['drug'].isin(ace_drugs + arb_drugs)]
    df_ace_arb_long = process_medications(ace_arb_df, 'drug')

    # Vasopressors
    # print("Processing vasopressors...")
    df_vaso = pd.read_csv(PATH_DATA / "VASO.csv", dtype={'subject_id': str, 'stay_id': str})
    df_events = pd.read_csv(PATH_DATA / "vasopressors.csv", dtype={'stay_id': str})
    df_events = df_events.merge(df_vaso[['stay_id', 'subject_id']], on='stay_id', how='left')
    df_events['drug'] = df_events['label'].astype(str).str.strip().str.title()
    df_events['amount'] = pd.to_numeric(df_events['amount'], errors='coerce')
    df_vaso_long = process_medications(df_events, 'drug', 'amount')

    # --- Combine ---
    # print("Combining all medications...")
    df_all_long = pd.concat([df_ab_long, df_diu_long, df_ace_arb_long, df_vaso_long], ignore_index=True)
    df_all_long = df_all_long.sort_values(['subject_id', 'starttime'])

    # Add AKI times
    df_all_long = df_all_long.merge(
        aki_times[['subject_id', 'AKI_time']],
        on='subject_id',
        how='left')

    # Filter 12h window before AKI
    df_all_long = df_all_long[
        (df_all_long['starttime'] >= df_all_long['AKI_time'] - pd.Timedelta(hours=12)) &
        (df_all_long['starttime'] <= df_all_long['AKI_time'])]

    # Drop exact duplicates
    df_all_long = df_all_long.drop_duplicates(subset=['subject_id', 'medication', 'starttime', 'AKI_time'])

    # --- Split into amounts vs binary ---
    df_amounts = df_all_long[df_all_long['amount'].notna()].copy()
    df_amounts['amount'] = df_amounts['amount'].round(2)
    amounts_path = REPO_ROOT / "csv_dashboard" / "meds_12h_before_AKI_amounts.csv"
    amounts_path.parent.mkdir(exist_ok=True)
    df_amounts.to_csv(amounts_path, index=False, decimal=',')

    df_binary = df_all_long[df_all_long['amount'].isna()].copy()
    df_binary['amount'] = 1  # binary = 1
    binary_path = REPO_ROOT / "csv_dashboard" / "meds_12h_before_AKI_binary.csv"
    df_binary.to_csv(binary_path, index=False, decimal=',')
    return df_binary,df_amounts
    # print(f"Amounts file saved to {amounts_path}")
    # print(f"Binary file saved to {binary_path}")


# if __name__ == "__main__":
#     combine_all_to_one_csv_long_with_time()
