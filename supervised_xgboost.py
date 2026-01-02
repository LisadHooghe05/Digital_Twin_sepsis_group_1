import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from xgboost import XGBClassifier, XGBRegressor

import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from unsupervised_clustering import cluster_analysis

REPO_ROOT = Path(__file__).resolve().parent
PATH_DATA = REPO_ROOT / "matrix_filled.csv"

SELECTED_FEATURES = [
    "age_12h_before_AKI",
    "Diastolic Blood Pressure",
    "Heart Rate",
    "Mean Arterial Pressure",
    "Oxygen Saturation",
    "Furosemide",
    "Vancomycin",
    "Cefepime",
    "Metoprolol",
    "Invasive ventilation",
    "Diabetes Mellitus",
    "Heart Failure",
    "Hypertension",
]

def load_matrix(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    elif path.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

def prepare_xy(df: pd.DataFrame, features: list[str], target_col: str):
    missing_cols = [c for c in features + [target_col] if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in df: {missing_cols}")

    X = df[features].copy()
    y = df[target_col].copy()

    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]

    return X, y

# ---------------------------
# 1) XGBoost (zoals je had)
# ---------------------------
def run_xgb_mortality(df: pd.DataFrame, target_col: str = "died_within_90d_after_AKI"):
    X, y = prepare_xy(df, SELECTED_FEATURES, target_col)
    y = y.astype(int)

    model = XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        eval_metric="logloss", tree_method="hist", random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Let op: dit wordt NaN als een fold maar 1 klasse heeft
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    pr_scores  = cross_val_score(model, X, y, cv=cv, scoring="average_precision")

    print(f"[{target_col}] CV ROC-AUC: {np.nanmean(auc_scores):.3f} ± {np.nanstd(auc_scores):.3f}")
    print(f"[{target_col}] CV PR-AUC : {np.nanmean(pr_scores):.3f} ± {np.nanstd(pr_scores):.3f}")

    model.fit(X, y)
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nXGBoost feature importances:")
    print(imp)

    return model, imp

def run_xgb_icu_stay(df: pd.DataFrame, target_col: str = "ICU_stay_12hforakitime"):
    X, y = prepare_xy(df, SELECTED_FEATURES, target_col)

    model = XGBRegressor(
        n_estimators=800, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        objective="reg:squarederror", tree_method="hist", random_state=42
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_scores = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
    r2_scores  = cross_val_score(model, X, y, cv=cv, scoring="r2")

    print(f"[{target_col}] CV MAE: {mae_scores.mean():.3f} ± {mae_scores.std():.3f}")
    print(f"[{target_col}] CV R2 : {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")

    model.fit(X, y)
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nXGBoost feature importances:")
    print(imp)

    return model, imp

# ----------------------------------------
# 2) Associaties: Logistic + OLS (+ FDR)
# ----------------------------------------
def mortality_associations(
    df: pd.DataFrame,
    target_col: str = "died_within_90d_after_AKI",
    fdr_alpha: float = 0.05
) -> pd.DataFrame:
    X, y = prepare_xy(df, SELECTED_FEATURES, target_col)
    y = y.astype(int)

    # Zorg dat binaire features echt 0/1 zijn (veilig)
    for col in ["Furosemide", "Invasive ventilation", "Diabetes Mellitus", "Heart Failure", "Hypertension"]:
        if col in X.columns:
            X[col] = X[col].fillna(0)
            X[col] = (X[col] > 0).astype(int)

    # Impute overige missing (consistent met jouw matrix_filled)
    X = X.fillna(X.median(numeric_only=True))

    X_sm = sm.add_constant(X, has_constant="add")

    # Logistische regressie (multivariaat)
    fit = sm.Logit(y, X_sm).fit(disp=0)

    res = pd.DataFrame({
        "feature": fit.params.index,
        "coef": fit.params.values,
        "p_value": fit.pvalues.values,
    })

    # Drop intercept
    res = res[res["feature"] != "const"].copy()

    res["odds_ratio"] = np.exp(res["coef"])

    # FDR-correctie
    _, q, _, _ = multipletests(res["p_value"].values, alpha=fdr_alpha, method="fdr_bh")
    res["q_value_fdr_bh"] = q
    res["significant_fdr"] = res["q_value_fdr_bh"] < fdr_alpha

    # sort
    res = res.sort_values(["q_value_fdr_bh", "p_value"])

    return res

def icu_associations(
    df: pd.DataFrame,
    target_col: str = "ICU_stay_12hforakitime",
    fdr_alpha: float = 0.05,
    log_transform: bool = True
) -> pd.DataFrame:
    X, y = prepare_xy(df, SELECTED_FEATURES, target_col)

    # binaire kolommen fixen (veilig)
    for col in ["Furosemide", "Invasive ventilation", "Diabetes Mellitus", "Heart Failure", "Hypertension"]:
        if col in X.columns:
            X[col] = X[col].fillna(0)
            X[col] = (X[col] > 0).astype(int)

    X = X.fillna(X.median(numeric_only=True))

    # log1p is vaak beter voor LOS
    y_model = np.log1p(y) if log_transform else y

    X_sm = sm.add_constant(X, has_constant="add")

    # OLS met robuuste standaardfouten (HC3)
    fit = sm.OLS(y_model, X_sm).fit(cov_type="HC3")

    res = pd.DataFrame({
        "feature": fit.params.index,
        "coef": fit.params.values,
        "p_value": fit.pvalues.values,
    })
    res = res[res["feature"] != "const"].copy()

    # Als log_transform: omzetting naar %-effect approx voor interpretatie
    if log_transform:
        res["approx_pct_change"] = (np.exp(res["coef"]) - 1.0) * 100.0

    _, q, _, _ = multipletests(res["p_value"].values, alpha=fdr_alpha, method="fdr_bh")
    res["q_value_fdr_bh"] = q
    res["significant_fdr"] = res["q_value_fdr_bh"] < fdr_alpha

    res = res.sort_values(["q_value_fdr_bh", "p_value"])
    return res

def supervised_and_associations():
    df_core, *_ = cluster_analysis(PATH_DATA, save_models=False)
    df = df_core

    # XGBoost (predictive)
    print("\n=== XGBoost: Mortality ===")
    run_xgb_mortality(df, target_col="died_within_90d_after_AKI")

    print("\n=== XGBoost: ICU stay ===")
    run_xgb_icu_stay(df, target_col="ICU_stay_12hforakitime")

    # Associations (explainable)
    print("\n=== Associations: Mortality (Logistic regression) ===")
    mort = mortality_associations(df, target_col="died_within_90d_after_AKI")
    print(mort[["feature", "odds_ratio", "p_value", "q_value_fdr_bh", "significant_fdr"]])

    print("\n=== Associations: ICU stay (OLS on log1p) ===")
    icu = icu_associations(df, target_col="ICU_stay_12hforakitime", log_transform=True)
    cols = ["feature", "coef", "p_value", "q_value_fdr_bh", "significant_fdr"]
    if "approx_pct_change" in icu.columns:
        cols.insert(2, "approx_pct_change")
    print(icu[cols])

if __name__ == "__main__":
    supervised_and_associations()
