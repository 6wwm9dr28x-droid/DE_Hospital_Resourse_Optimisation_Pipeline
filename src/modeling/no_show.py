import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List, Tuple

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay

# Make statsmodels optional (avoid hard failure if incompatible with SciPy)
try:
    import statsmodels.api as sm
    HAVE_SM = True
except Exception:
    HAVE_SM = False

from src.utils.paths import PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, ensure_dirs

# -----------------------------
# Utilities / feature selection
# -----------------------------

def load_kaggle_clean() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / "kaggle_clean.csv")

def feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], str]:
    target = "no_show_flag"
    numeric_candidates = [
        "waitingdays", "age", "scholarship", "sms_received",
        "hipertension", "hypertension", "diabetes", "alcoholism", "handcap"
    ]
    num_cols = [c for c in numeric_candidates if c in df.columns]
    cat_candidates = ["gender", "neighbourhood"]
    cat_cols = [c for c in cat_candidates if c in df.columns]
    return num_cols, cat_cols, target

# -----------------------------
# RQ2_Fig1: WaitingDays vs No-show
# -----------------------------

def make_waitingdays_plot(df: pd.DataFrame) -> Path:
    tmp = df.copy()
    tmp = tmp[~tmp["waitingdays"].isna()]
    tmp["no_show_flag"] = tmp["no_show_flag"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(8, 4))
    showed = tmp.loc[tmp["no_show_flag"] == 0, "waitingdays"].astype(float)
    noshow = tmp.loc[tmp["no_show_flag"] == 1, "waitingdays"].astype(float)

    bins = np.linspace(tmp["waitingdays"].min(), tmp["waitingdays"].max(), 50)
    ax.hist(showed, bins=bins, alpha=0.6, label="Show", color="#2b8a3e")
    ax.hist(noshow, bins=bins, alpha=0.6, label="No-show", color="#e8590c")

    ax.set_title("WaitingDays distribution by attendance outcome")
    ax.set_xlabel("WaitingDays (days)")
    ax.set_ylabel("Count")
    ax.legend()

    out_path = FIGURES_DIR / "RQ2_Fig1.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out_path

# -----------------------------------------------------------
# RQ2_Fig2: ROC curve + Confusion matrix (Logistic baseline)
# -----------------------------------------------------------

def train_logistic_and_metrics(df: pd.DataFrame):
    num_cols, cat_cols, target = feature_columns(df)
    feat_cols = num_cols + cat_cols

    df2 = df.dropna(subset=[target], how="any").copy()
    df2 = df2.dropna(subset=feat_cols, how="all")
    X = df2[feat_cols].copy()
    y = df2[target].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    # scikit-learn >=1.2 uses penalty=None to mean 'no regularization'
    logit = LogisticRegression(penalty=None, max_iter=500, solver="lbfgs")

    pipe = Pipeline(steps=[("pre", pre), ("clf", logit)])
    pipe.fit(X_train, y_train)

    y_score = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_score)
    y_pred = (y_score >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    RocCurveDisplay.from_predictions(y_test, y_score, ax=axes[0])
    axes[0].set_title(f"ROC curve (AUC = {auc:.3f})")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=axes[1], cmap="Blues", colorbar=False)
    axes[1].set_title("Confusion matrix @ threshold = 0.5")

    out_path = FIGURES_DIR / "RQ2_Fig2.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    return pipe, out_path, auc

# ------------------------------------------------------------
# RQ2_Table1: Coefficients, Odds Ratios, Confidence Intervals
# ------------------------------------------------------------

def make_coeff_table(pipe: Pipeline, df: pd.DataFrame) -> Path:
    num_cols, cat_cols, target = feature_columns(df)
    feat_cols = num_cols + cat_cols

    df2 = df.dropna(subset=[target], how="any").copy()
    df2 = df2.dropna(subset=feat_cols, how="all")
    X = df2[feat_cols].copy()
    y = df2[target].astype(int).values

    pre: ColumnTransformer = pipe.named_steps["pre"]
    names_num = list(pre.transformers_[0][2]) if num_cols else []
    ohe: OneHotEncoder = pre.transformers_[1][1] if cat_cols else None
    names_cat = list(ohe.get_feature_names_out(cat_cols)) if cat_cols else []
    feature_names = names_num + names_cat

    clf: LogisticRegression = pipe.named_steps["clf"]
    coef = clf.coef_.ravel()

    df_coef = pd.DataFrame({
        "feature": feature_names,
        "coef_logit": coef,
        "odds_ratio": np.exp(coef)
    })

    # Optional CIs via statsmodels if available and compatible
    if HAVE_SM:
        try:
            X_trans = pre.transform(X)  # use fitted preprocessor
            X_sm = sm.add_constant(X_trans)
            model = sm.Logit(y, X_sm, missing="drop")
            res = model.fit(disp=False)
            ci = res.conf_int()
            sm_names = ["const"] + feature_names
            ci_df = pd.DataFrame({"feature": sm_names, "ci_low": ci[0], "ci_high": ci[1]})
            ci_df = ci_df[ci_df["feature"] != "const"]
            df_coef = df_coef.merge(ci_df, on="feature", how="left")
        except Exception:
            df_coef["ci_low"] = np.nan
            df_coef["ci_high"] = np.nan
    else:
        df_coef["ci_low"] = np.nan
        df_coef["ci_high"] = np.nan

    # Threshold metrics @ 0.5
    y_score = pipe.predict_proba(X)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / cm.sum()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    metrics_df = pd.DataFrame({"metric": ["accuracy", "precision", "recall"],
                               "value": [acc, precision, recall]})

    out_path = TABLES_DIR / "RQ2_Table1.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_coef.sort_values("odds_ratio", ascending=False).to_excel(writer, index=False, sheet_name="Coefficients_OR_CI")
        metrics_df.to_excel(writer, index=False, sheet_name="Threshold_Metrics")
    return out_path

# -------------------------------------------------------------
# RQ2_Table2: No-show rate by neighbourhood and age band
# -------------------------------------------------------------

def make_cohort_table(df: pd.DataFrame) -> Path:
    tmp = df.copy()

    # Age bands (categorical)
    if "age" in tmp.columns:
        bins   = [0, 12, 18, 30, 45, 60, 75, 90, 120]
        labels = ["0-11", "12-17", "18-29", "30-44", "45-59", "60-74", "75-89", "90+"]
        tmp["age_band"] = pd.cut(tmp["age"], bins=bins, labels=labels, right=False)
        # Keep only observed bands, and replace NaN with 'unknown'
        tmp["age_band"] = tmp["age_band"].cat.remove_unused_categories()
        tmp["age_band"] = tmp["age_band"].cat.add_categories(["unknown"]).fillna("unknown")
    else:
        tmp["age_band"] = "unknown"

    # Neighbourhood cleanup
    if "neighbourhood" not in tmp.columns:
        tmp["neighbourhood"] = "unknown"
    tmp["neighbourhood"] = tmp["neighbourhood"].fillna("unknown")

    # Target cleanup
    tmp["no_show_flag"] = tmp["no_show_flag"].fillna(0.0)

    # Group using only observed combinations; this prevents length-mismatch on reinsertion
    grp = tmp.groupby(["neighbourhood", "age_band"], observed=True, as_index=False).agg(
        count=("no_show_flag", "size"),
        no_show_rate=("no_show_flag", "mean")
    )
    grp["no_show_rate_pct"] = (grp["no_show_rate"] * 100).round(2)

    out_path = TABLES_DIR / "RQ2_Table2.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        grp.sort_values(["no_show_rate", "count"], ascending=[False, False]).to_excel(
            writer, index=False, sheet_name="NoShow_By_Cohort"
        )
    return out_path

# -----------------------------
# Driver
# -----------------------------

def main():
    ensure_dirs()
    df = load_kaggle_clean()

    fig1 = make_waitingdays_plot(df)
    print(f"✅ Saved: {fig1}")

    pipe, fig2, auc = train_logistic_and_metrics(df)
    print(f"✅ Saved: {fig2} (AUC={auc:.3f})")

    t1 = make_coeff_table(pipe, df)
    print(f"✅ Saved: {t1}")

    t2 = make_cohort_table(df)
    print(f"✅ Saved: {t2}")

    print("\nRQ2 artifacts generated successfully.")
