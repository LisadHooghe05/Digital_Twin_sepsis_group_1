import pandas as pd
from pathlib import Path

# Set root and file path
REPO_ROOT = Path(__file__).resolve().parent
file_path = REPO_ROOT / "matrix_filled.csv"



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
    df_dashboard = df_core[['subject_id', 'cluster', 'cluster_prob', 'Autoimmune / Vasculitis', 
                            'Chronic Kidney Disease', 'Diabetes Mellitus', 'Heart Failure',
                            'Hypertension', 'Malignancy', 'Obstructive Uropathy',
                            'Sepsis', 'Silhouette_score', 'HDBSCAN_proba']].copy()
    
    # Add Silhouette Color 
    df_dashboard['Silhouette_Color'] = df_dashboard['Silhouette_score'].apply(
        lambda x: None if pd.isna(x) else ("#FF0000" if x < 0.7 else "#0000FF"))

    # Add Cluster Color 
    df_dashboard['Cluster_Color'] = df_dashboard['cluster_prob'].apply(
        lambda x: None if pd.isna(x) else ("#FF0000" if x < 0.7 else "#0000FF"))

    out_dir = REPO_ROOT / "csv_dashboard"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_dashboard.to_csv(out_dir / "df_dashboard.csv", index=False)

    return df_dashboard

# if __name__ == "__main__":
#     #Run clustering
#     df_core, bic_scores, sil, dbi, kw_df, mortality_rates, vt, scaler, pca, best_gmm = cluster_analysis(
#         file_path,
#         variance_thresh=0.01,
#         pca_variance=0.90,
#         min_cluster_size=50,
#         hdb_prob_thresh=0.848

