from determining_AKI import AKI_detection
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT_dist = Path(__file__).resolve().parent
PATH_DATA_dist = REPO_ROOT_dist / "data"
SEPSIS_CSV = "sepsis_diagnose_time.csv"
CREAT_CSV = "creatinine_over_time.csv"


def sepsis_AKI_time():
    df = AKI_detection(SEPSIS_CSV, CREAT_CSV).copy()
    df['days_to_AKI'] = pd.to_numeric(df['days_to_AKI'], errors='coerce')
    df_valid = df[(df['days_to_AKI'] >= 0) & (df['days_to_AKI'] <= 7)].copy()

    plt.figure(figsize=(8, 5))

    values = df_valid['days_to_AKI'].to_numpy()
    counts, edges = np.histogram(values, bins=28, range=(0, 7))
    centers = (edges[:-1] + edges[1:]) / 2.0

    
    # plt.figure(figsize=(8, 5))
    # plt.plot(centers, counts, marker='o')  # lijn i.p.v. bars
    # plt.xlabel("Tijd tot AKI (dagen na sepsis)")
    # plt.ylabel("Aantal patiënten")
    # plt.title("Tijd tot optreden van AKI na sepsis (0–7 dagen)")
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.xlim(0, 7)  # forceer x-as naar 7 dagen
    # plt.tight_layout()
    # plt.show()


    
    return df_valid[['subject_id', 'sepsis_time', 'days_to_AKI']]

