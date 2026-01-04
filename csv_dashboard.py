import pandas as pd
from pathlib import Path
import numpy as np

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
                            'Sepsis', 'Silhouette_score', 'HDBSCAN_proba', 'Invasive ventilation', 'Cefepime',
                            'age_12h_before_AKI', 'Metoprolol', 'Furosemide (Lasix)', 'Furosemide (Lasix) 250/50',
                            'Vancomycin']].copy()
    
    # Add Silhouette Color 
    df_dashboard['Silhouette_Color'] = df_dashboard['Silhouette_score'].apply(
        lambda x: None if pd.isna(x) else ("#FF0000" if x < 0.7 else "#0000FF"))

    # Add Cluster Color 
    df_dashboard['Cluster_Color'] = df_dashboard['cluster_prob'].apply(
        lambda x: None if pd.isna(x) else ("#FF0000" if x < 0.7 else "#0000FF"))

    # Add Medical Advice
    cluster_advice_map = {3: "Cluster 3 shows the lowest mortality (13.9%), and a short ICU stay (~2d). Patients display more stable physiology and fewer high-risk features, making this the most favorable phenotype.",
                          4: "Cluster 4 shows the highest mortality (18.1%) and the moderate ICU stay (~5.7d). This phenoty concentrates more high-risk features and represents the least favorable survival profile.",
                          5: "Cluster 5 shows moderate mortality (14.9%) but the longest ICU stay (~6.7d). This phenotype suggests a prolonged recovery trajectory despite not having the highest morality."}   
    df_dashboard['Cluster_Advice'] = df_dashboard['cluster'].map(cluster_advice_map)

    feature_advice_map = {'Heart Failure': {'condition': lambda x: x == 1,
                                            'text': "History of heart failure, strongly associated with higher mortality risk."},
                          'Diabetes Mellitus': {'condition': lambda x: x == 1,
                                            'text': "History of diabetes, associated with elevated mortality risk."},
                          'Hypertension': {'condition': lambda x: x == 1,
                                            'text': "History of hypertension, associated with longer ICU stay."},
                          'Invasive ventilation': {'condition': lambda x: x > 0,
                                            'text': "Invasive ventilation shows clear association with increased mortality odds."},
                          'Cefepime': {'condition': lambda x: x > 0,
                                            'text': "Cefepime us is linked to higher mortality odds."},
                          # Everybody will have this advice
                          'age_12h_before_AKI': {'condition': lambda x: x > 0,
                                            'text': "Age shows positive association with mortality odds in the model."},
                          'Metoprolol': {'condition': lambda x: x > 0,
                                            'text': "Shows a modest association with increased mortality odds."},
                          'Furosemide (Lasix)': {'condition': lambda x: x > 0,
                                            'text': "Furosemide is associated with a shorter ICU stay."},
                          'Furosemide (Lasix) 250/50': {'condition': lambda x: x > 0,
                                            'text': "Furosemide is associated with a shorter ICU stay."},
                          'Vancomycin': {'condition': lambda x: x > 0,
                                            'text': "Vancomycin is linked to a reduced ICU stay duration."}}
    
    for feature, rule in feature_advice_map.items():
        if feature in df_dashboard.columns:
            mask = df_dashboard[feature].apply(rule['condition'])
            df_dashboard.loc[mask, f'{feature}_Advice'] = rule['text']

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

