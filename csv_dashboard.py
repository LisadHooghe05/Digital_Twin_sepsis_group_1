import pandas as pd
from pathlib import Path
from unsupervised_clustering import cluster_analysis

# Set root and file path
REPO_ROOT = Path(__file__).resolve().parent
file_path = REPO_ROOT / "matrix_final_final.csv"

# Run clustering
df_core, bic_scores, sil, dbi, kw_df, mortality_rates, vt, scaler, pca, best_gmm = cluster_analysis(
    file_path,
    variance_thresh=0.01,
    pca_variance=0.90,
    min_cluster_size=50,
    hdb_prob_thresh=0.848
)

def dataframe_dashboard(df_core):
    """
    Assemble dataframe used for clinical dashboard.

    Parameters
    ----------
    df_core : pd.DataFrame
        Dataframe with patients clustered by GMM clustering.
    
    Returns
    -------
    pd.DataFrame
        Dashboard dataframe with selected patient info and clustering scores.
    """
    df_dashboard = df_core[['subject_id', 'cluster', 'Autoimmune / Vasculitis', 
                            'Chronic Kidney Disease', 'Diabetes Mellitus', 'Heart Failure',
                            'Hypertension', 'Malignancy', 'Obstructive Uropathy',
                            'Sepsis', 'Silhouette_score', 'HDBSCAN_proba']].copy()
    return df_dashboard

# Create dashboard dataframe
df_dashboard = dataframe_dashboard(df_core)
#print(df_dashboard.head(10))
