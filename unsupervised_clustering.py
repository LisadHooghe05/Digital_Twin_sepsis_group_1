
import pandas as pd
import hdbscan
from hdbscan.prediction import approximate_predict
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp
import itertools
from joblib import dump, load
from csv_dashboard import dataframe_dashboard

# Set root
REPO_ROOT = Path(__file__).resolve().parent
PATH_DATA = REPO_ROOT / "matrix_filled.csv"

def cluster_analysis(file_path, variance_thresh=0.01, pca_variance=0.90, 
                     min_cluster_size=50, hdb_prob_thresh=0.848, save_models=True):
    """
    Perform HDBSCAN + GMM clustering, calculate mortality, and compute feature importance.

    Returns:
        df_core: DataFrame with cluster labels, silhouette scores, and HDBSCAN probability
        bic_scores: list of BIC scores for k=3..7
        sil: overall silhouette score
        dbi: Davies-Bouldin index
        kw_df: Kruskal-Wallis statistics per feature
        mortality_rates: mean mortality per cluster
        vt, scaler, pca, best_gmm: fitted objects for transforming new patients
    """
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Columns to exclude from clustering
    exclude_cols = ["subject_id", "ICU_stay_12hforakitime", "died_within_90d_after_AKI"]
    X = df.drop(columns=exclude_cols)
    
    # Remove near-zero variance
    vt = VarianceThreshold(threshold=variance_thresh)
    X_var = vt.fit_transform(X)
    
    # Scale data
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_var)
    
    # PCA
    pca = PCA(n_components=pca_variance)
    X_pca = pca.fit_transform(X_scaled)
    
    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True)
    hdb_labels = clusterer.fit_predict(X_pca)
    proba = hdbscan.all_points_membership_vectors(clusterer)
    max_proba = proba.max(axis=1)
    
    # Select core points
    core_mask = (hdb_labels >= 0) & (max_proba >= hdb_prob_thresh)
    X_core = X_pca[core_mask]
    df_core = df[core_mask].copy()
    print(f"HDBSCAN: "
    f"{(hdb_labels >= 0).sum()} non-noise, "
    f"{(max_proba >= hdb_prob_thresh).sum()} above prob thresh, "
    f"{core_mask.sum()} core points")
    
    # GMM clustering
    bic_scores, models = [], []
    for k in range(3, 8):
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42,
                              reg_covar=1e-6, n_init=10)
        gmm.fit(X_core)
        bic_scores.append(gmm.bic(X_core))
        models.append(gmm)
    
    best_k = np.argmin(bic_scores) + 3
    best_gmm = models[np.argmin(bic_scores)]
    print(f"best number cluster: {best_k}")
    
    labels = best_gmm.predict(X_core)

    core_probs = best_gmm.predict_proba(X_core).max(axis=1)
    df_core['cluster_prob'] = core_probs


    df_core['cluster'] = labels
    df_core['HDBSCAN_proba'] = max_proba[core_mask]
    
    # Add only the significant clusters in df_core
    MIN_CLUSTER_SIZE = 70
    cluster_sizes = df_core['cluster'].value_counts()
    valid_clusters = cluster_sizes[cluster_sizes >= MIN_CLUSTER_SIZE].index.tolist()
    
    valid_mask = df_core['cluster'].isin(valid_clusters).values  # lengte = len(df_core) = len(X_core)

    df_core = df_core.loc[valid_mask].copy()
    X_core = X_core[valid_mask, :]          

    X_core_filtered = X_core
    labels_filtered = df_core['cluster'].values

    
    # Define cluster distribution for cluster analysis
    cluster_counts = df_core['cluster'].value_counts().sort_index()
    cluster_percentages = (cluster_counts / cluster_counts.sum()) * 100
    cluster_distribution = pd.DataFrame({"n_patients": cluster_counts,
                                         "percentage": cluster_percentages})
    
    # Silhouette and Davies-Bouldin
    if len(np.unique(labels)) > 1:
        sil = silhouette_score(X_core_filtered, labels_filtered)
        sil_per_patient = silhouette_samples(X_core_filtered, labels_filtered)

        df_core['Silhouette_score'] = sil_per_patient

        dbi = davies_bouldin_score(X_core_filtered, labels_filtered)
    else:
        sil = dbi = np.nan
        df_core['Silhouette_score'] = np.nan

    
    # Mortality rates per cluster
    mortality_rates = df_core.groupby('cluster')['died_within_90d_after_AKI'].mean()
    # ICU stay per cluster
    icu_stay_rates = (df_core.loc[df_core['ICU_stay_12hforakitime'] != 0].groupby('cluster')['ICU_stay_12hforakitime'].mean())
    
    # Kruskal-Wallis and eta-squared feature importance
    eta_sq_threshold = 0.01
    rows = []

    # Identify valid clusters
    cluster_sizes = df_core['cluster'].value_counts()
    valid_clusters = cluster_sizes[cluster_sizes >= MIN_CLUSTER_SIZE].index.tolist()

    if len(valid_clusters) < 2:
        raise ValueError("Not enough clusters ≥ min_cluster_size for Kruskal-Wallis test")

    feature_cols = df_core.columns.drop(['cluster',
        'ICU_stay_12hforakitime','died_within_90d_after_AKI',
        'cluster_prob', 'HDBSCAN_proba', 'Silhouette_score'])

    for col in feature_cols:
        groups = [df_core.loc[df_core['cluster'] == c, col].dropna()
            for c in valid_clusters]

        if any(g.nunique() <= 1 for g in groups):
            continue

        stat, p = kruskal(*groups)

        n = sum(len(g) for g in groups)
        k = len(groups)

        eta_sq = (stat - k + 1) / (n - k)

        if eta_sq < eta_sq_threshold:
            continue
        rows.append({"feature": col, "H": stat, "p": p,
                     "eta_sq": eta_sq})
    
    kw_df = pd.DataFrame(rows)
    
    # Save models for later new patient assignment
    if save_models:
        dump(vt, REPO_ROOT / "vt_model.joblib")
        dump(scaler, REPO_ROOT / "scaler_model.joblib")
        dump(pca, REPO_ROOT / "pca_model.joblib")
        dump(best_gmm, REPO_ROOT / "gmm_model.joblib")
        dump(X_core, REPO_ROOT /  "X_core_pca.joblib")      
        dump(labels_filtered, REPO_ROOT / "labels_core.joblib")
        dump(clusterer, REPO_ROOT / "hdbscan_model.joblib")

    out_dir = REPO_ROOT / "csv_dashboard"
    out_dir.mkdir(exist_ok=True)

    kw_df.to_csv(out_dir / "cluster_feature_importance.csv", index=False)
  
    return df_core, bic_scores, sil, dbi, kw_df, cluster_distribution, mortality_rates, icu_stay_rates, vt, scaler, pca, best_gmm


def comparing_clusters(cluster_df, significance_df):
    """
    Analyze important features using means, FDR correction, and Dunn test for pairwise clusters.
    """
    important_features = ["Oxygen Saturation", "Furosemide", "Vancomycin",
                          "Norepinephrine", "Heart Failure"]
    
    mean = cluster_df.groupby("cluster")[important_features].mean(numeric_only=True)

    # FDR adjustment
    mask = significance_df["p"].notna()
    significance_df.loc[mask, "p_fdr"] = multipletests(significance_df.loc[mask, "p"], method="fdr_bh")[1]
    significance_df["consider"] = (significance_df["p_fdr"] < 0.05) & (significance_df["eta_sq"] > 0.06)
    significance_df = significance_df.sort_values(["consider", "p_fdr"], ascending=[False, True])

    # Dunn test for pairwise cluster comparisons
    all_rows = []
    for feature in important_features:
        p_mat = sp.posthoc_dunn(cluster_df, val_col=feature, group_col="cluster", p_adjust="holm")
        for a, b in itertools.combinations(p_mat.index, 2):
            p = float(p_mat.loc[a, b])
            all_rows.append({
                "feature": feature,
                "cluster_a": a,
                "cluster_b": b,
                "p_adj": p,
                "significant": p < 0.05,
                "mean_a": mean.loc[a, feature],
                "mean_b": mean.loc[b, feature],
                "diff_b_minus_a": mean.loc[b, feature] - mean.loc[a, feature]
            })
    dunn_output = pd.DataFrame(all_rows).sort_values(["feature", "p_adj"])
    
    return mean, significance_df, dunn_output


def assign_patient(patient_feature_df, df_core):
    """
    Assign a new patient to an existing cluster. (Process patient exactly the same way as 
    overall clustering)
    
    patient_df: single-row dataframe with the same features used for clustering
    df_core: dataframe returned from cluster_analysis
    """
    # Force vital input or median from df_core
    vitals = {'age_12h_before_AKI': 68,
              'Diastolic Blood Pressure': 61,
              'Heart Rate': 88.18,
              'Mean Arterial Pressure': 73.10,
              'Oxygen Saturation': 96.77,
              'Respiratory Rate': 20.17,
                'Systolic Blood Pressure':110}
    for vital, median in vitals.items():
        if vital in patient_feature_df.columns:
            patient_feature_df[vital] = patient_feature_df[vital].fillna(median)
    
    # Add 0 if feature not defined
    other_features = []
    for column in patient_feature_df.columns:
        if column not in vitals:
            other_features.append(column)
    for feature in other_features:
        patient_feature_df[feature] = patient_feature_df[feature].fillna(0)
        
    # Load clustering modelling
    vt = load(REPO_ROOT / 'vt_model.joblib')
    scaler = load(REPO_ROOT / 'scaler_model.joblib')
    pca = load(REPO_ROOT / 'pca_model.joblib')
    gmm = load(REPO_ROOT / 'gmm_model.joblib')
    X_core = load(REPO_ROOT / 'X_core_pca.joblib')
    labels = load(REPO_ROOT / 'labels_core.joblib')
    clusterer = load(REPO_ROOT / "hdbscan_model.joblib")

    patient_X = patient_feature_df.drop(columns=['subject_id', 'ICU_stay_12hforakitime', 'died_within_90d_after_AKI'], errors='ignore')

    # Perform clustering preprocessing
    X_var = vt.transform(patient_X)
    X_scaled = scaler.transform(X_var)
    X_pca = pca.transform(X_scaled)
    
    # Predict cluster
    cluster_label = gmm.predict(X_pca)[0]
    
    # Cluster probability
    cluster_prob = gmm.predict_proba(X_pca).max()
    
    # Compute silhouette of new patient
    all_X = np.vstack([X_core, X_pca])
    all_labels = np.append(np.array(labels).ravel(), cluster_label)
    sil_values = silhouette_samples(all_X, all_labels)  
    # New patient is last value
    sil_score = sil_values[-1]

    # Predict cluster and membership probability for the new patient
    hdbscan_label, hdbscan_proba = approximate_predict(clusterer, X_pca)
    hdbscan_proba = hdbscan_proba[0]
    
    # Update patient row
    patient_feature_df_extended = patient_feature_df.copy().assign(cluster=cluster_label, cluster_prob=cluster_prob, 
                                                                   HDBSCAN_proba=hdbscan_proba, Silhouette_score=sil_score)

    df_core_dashboard = pd.concat([df_core, patient_feature_df_extended], ignore_index=True).copy()
    
    # Process new dashboard csv
    df_dashboard = dataframe_dashboard(df_core_dashboard)
    
    # Overwrite older df_dashboard csv
    out_dir = REPO_ROOT / "csv_dashboard"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_dashboard.to_csv(out_dir / "df_dashboard.csv", index=False)
    
    return df_dashboard


if __name__ == "__main__":
    df_core, bic_scores, sil, dbi, kw_df, cluster_distribution, mortality_rates, icu_stay_rates, vt, scaler, pca, best_gmm = cluster_analysis(PATH_DATA,variance_thresh=0.01, pca_variance=0.90, 
                     min_cluster_size=50, hdb_prob_thresh=0.835, save_models=True)
    print(f" bic: {bic_scores}, dbi: {dbi}, sil: {sil}, mortality rate:{mortality_rates}, ICU stay lengths:{icu_stay_rates}")
    
    mean, significance_df, dunn_output = comparing_clusters(df_core, kw_df)
    pd.set_option('display.max_rows', None)
    #print(dunn_output)
    
    
