import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def safe_numeric_sum(df, cols):
    return (
        df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .sum(axis=1)
    )

def safe_numeric_mean(df, cols):
    return (
        df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .mean(axis=1)
    )

# -----------------------------
# CONFIG
# -----------------------------
DEBUG = True
REBUILD_TARGET = True
TARGET_RATE = 0.10
THRESHOLD = 0.30


# -----------------------------
# HELPERS
# -----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    std = s.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def build_learnable_future_risk(df: pd.DataFrame, positive_rate_target: float = 0.10) -> pd.DataFrame:
    """
    Rebuild future risk target with stronger, learnable signal.
    Uses whichever supporting columns actually exist.
    """
    out = df.copy()

    def safe_series(col_name, default=0.0):
        if col_name in out.columns:
            return out[col_name].fillna(default).astype(float)
        return pd.Series(default, index=out.index, dtype="float64")

    # Core numeric drivers
    age = safe_series("age")
    pmpm = safe_series("pmpm_last_6m")
    util = safe_series("utilization_intensity")
    complexity = safe_series("complexity_score")
    rx_ratio = safe_series("rx_ratio")
    specialist_ratio = safe_series("specialist_ratio")
    acute = safe_series("acute_event_flag")
    ip_visits = safe_series("ip_visit_count")
    ed_visits = safe_series("ed_visit_count")
    dx_count = safe_series("distinct_dx1_count_last_6m")
    trend = safe_series("cost_trend_proxy")
    prior_high_cost = safe_series("high_cost_claim_flag")

    # Fallbacks if key drivers are missing
    if (pmpm == 0).all() and "total_cost_last_6m" in out.columns:
        total_cost = safe_series("total_cost_last_6m")
        months = 6.0
        pmpm = total_cost / months

    if (util == 0).all():
        util_parts = [c for c in ["medical_claim_count_last_6m", "pcp_visit_count", "specialist_visit_count"] if c in out.columns]
        if util_parts:
            print("util_parts:", util_parts)
            df["utilization_intensity"] = safe_numeric_sum(df, util_parts)
    else:
        df["utilization_intensity"] = 0

    complexity_cols = [c for c in df.columns if any(x in c.lower() for x in ["dx", "ndc", "risk"])]
    print("complexity_cols:", complexity_cols)

    if complexity_cols:
        print("complexity_cols:", complexity_cols)
        df["complexity_score"] = safe_numeric_sum(df, complexity_cols)
    else:
        df["complexity_score"] = 0

    # Standardize
    age_z = zscore(age)
    pmpm_z = zscore(pmpm)
    util_z = zscore(util)
    complexity_z = zscore(complexity)
    rx_z = zscore(rx_ratio)
    specialist_z = zscore(specialist_ratio)
    ip_z = zscore(ip_visits)
    ed_z = zscore(ed_visits)
    dx_z = zscore(dx_count)
    trend_z = zscore(trend)

    # Latent learnable risk
    latent_risk = (
        1.20 * pmpm_z
        + 0.90 * util_z
        + 0.75 * complexity_z
        + 0.60 * acute
        + 0.45 * age_z
        + 0.35 * rx_z
        + 0.25 * specialist_z
        + 0.80 * ip_z
        + 0.50 * ed_z
        + 0.40 * dx_z
        + 0.50 * trend_z
        + 0.90 * prior_high_cost
        + np.random.normal(0, 0.60, len(out))
    )

    # Calibrate prevalence
    cutoff = np.quantile(latent_risk, 1 - positive_rate_target)
    shifted_logit = latent_risk - cutoff
    prob_high_cost = sigmoid(shifted_logit * 1.35)

    rng = np.random.default_rng(42)
    out["high_cost_next_6m_flag"] = rng.binomial(1, prob_high_cost).astype(int)

    # Build future cost using available drivers
    base_future_cost = (
        pmpm * 4.8
        + util * 180
        + complexity * 350
        + acute * 2200
        + age * 18
    )

    high_cost_lift = rng.lognormal(mean=9.1, sigma=0.55, size=len(out))
    normal_cost_noise = rng.lognormal(mean=7.3, sigma=0.45, size=len(out))

    out["future_6m_total_cost"] = np.where(
        out["high_cost_next_6m_flag"] == 1,
        base_future_cost + high_cost_lift,
        base_future_cost * 0.65 + normal_cost_noise
    )

    out["future_6m_total_cost"] = out["future_6m_total_cost"].clip(lower=0).round(2)

    return out


# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("syn_model_snapshot.csv")
df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

print("Loaded shape:", df.shape)

if DEBUG:
    print("\nColumns at load:")
    print(df.columns.tolist())

    print("\nTarget rate by snapshot:")
    print(df.groupby("snapshot_date")["high_cost_next_6m_flag"].mean())


# -----------------------------
# REAL FEATURE ENGINEERING
# -----------------------------

# Utilization
util_cols = [c for c in df.columns if any(x in c.lower() for x in ["count", "visit", "ed", "readmit"])]
print("util_cols:", util_cols)

if util_cols:
    df["utilization_intensity"] = safe_numeric_sum(df, util_cols)
else:
    df["utilization_intensity"] = 0


# Complexity
complexity_cols = [c for c in df.columns if any(x in c.lower() for x in ["dx", "ndc", "risk"])]
print("complexity_cols:", complexity_cols)

if complexity_cols:
    df["complexity_score"] = safe_numeric_sum(df, complexity_cols)
else:
    df["complexity_score"] = 0


# Acute
acute_cols = [
    c for c in df.columns
    if any(x in c.lower() for x in ["ed_count", "readmit", "admit"])
]
print("acute_cols:", acute_cols)

if acute_cols:
    df["acute_event_flag"] = (safe_numeric_sum(df, acute_cols) > 0).astype(int)
else:
    df["acute_event_flag"] = 0


# Cost trend
cost_cols = [c for c in df.columns if any(x in c.lower() for x in ["cost", "allowed"])]
print("cost_cols:", cost_cols)

if cost_cols:
    df["cost_trend_proxy"] = safe_numeric_mean(df, cost_cols)
else:
    df["cost_trend_proxy"] = 0


# Final sanity check
print("\nFeature check:")
print(df[[
    "utilization_intensity",
    "complexity_score",
    "acute_event_flag",
    "cost_trend_proxy"
]].describe())


# Specialist ratio proxy (if no direct fields)
if "specialty_rx_count_3m" in df.columns:
    df["specialist_ratio"] = df["specialty_rx_count_3m"] / (df["utilization_intensity"] + 1)
else:
    df["specialist_ratio"] = 0


# 5. Cost × utilization
if "cost_intensity" in df.columns and "medical_claim_count_last_6m" in df.columns:
    df["cost_x_util"] = df["cost_intensity"] * df["medical_claim_count_last_6m"]
else:
    df["cost_x_util"] = 0.0

# 6. Pharmacy intensity
if "rx_cost_last_6m" in df.columns and "total_cost_last_6m" in df.columns:
    df["rx_ratio"] = df["rx_cost_last_6m"] / (df["total_cost_last_6m"] + 1)
else:
    df["rx_ratio"] = 0.0

# 7. Cost trend proxy
if "pmpm_last_6m" in df.columns and "benchmark_pmpm" in df.columns:
    df["cost_trend_proxy"] = df["pmpm_last_6m"] / (df["benchmark_pmpm"] + 1)
elif "pmpm_last_6m" in df.columns:
    df["cost_trend_proxy"] = df["pmpm_last_6m"]
else:
    df["cost_trend_proxy"] = 0.0

# 8. Risk interaction
if "risk_score" in df.columns and "total_cost_last_6m" in df.columns:
    df["risk_x_cost"] = df["risk_score"] * df["total_cost_last_6m"]
else:
    df["risk_x_cost"] = 0.0

# 9. Specialist-heavy mix
if "specialist_visit_count" in df.columns and "pcp_visit_count" in df.columns:
    df["specialist_ratio"] = df["specialist_visit_count"] / (df["pcp_visit_count"] + 1)
else:
    df["specialist_ratio"] = 0.0

# 10. Acute event flag
acute_parts = []
if "admit_count_last_6m" in df.columns:
    acute_parts.append(df["admit_count_last_6m"] > 0)
if "avoidable_ed_count_last_6m" in df.columns:
    acute_parts.append(df["avoidable_ed_count_last_6m"] > 0)

# -----------------------------
# 3. OPTIONAL TARGET REBUILD
# -----------------------------
if REBUILD_TARGET:
    df = build_learnable_future_risk(df, positive_rate_target=TARGET_RATE)

    print("\nRebuilt target and future cost with learnable signal.")
    print("New target distribution:")
    print(df["high_cost_next_6m_flag"].value_counts(normalize=True).sort_index())

    # Optional: save rebuilt dataset for reuse
    df.to_csv("syn_model_snapshot_rebuilt.csv", index=False)
    print("\nSaved rebuilt dataset: syn_model_snapshot_rebuilt.csv")


# -----------------------------
# 4. DEBUG SIGNAL CHECK
# -----------------------------
if DEBUG:
    candidate_cols = [
        "pmpm_last_6m",
        "utilization_intensity",
        "complexity_score",
        "acute_event_flag",
        "cost_trend_proxy",
        "distinct_dx1_count_last_6m"
    ]

    available_cols = [c for c in candidate_cols if c in df.columns]

    print("\nAvailable signal columns:", available_cols)

    if "high_cost_next_6m_flag" in df.columns and available_cols:
        signal_check = df.groupby("high_cost_next_6m_flag")[available_cols].mean()
        print("\nSignal check by target:")
        print(signal_check)
    else:
        print("\nCannot run signal check — missing target or expected feature columns.")

    print("\nDataset shape after feature engineering / target rebuild:", df.shape)
    print("\nTarget distribution:")
    print(df["high_cost_next_6m_flag"].value_counts())
    print("\nPreview:")
    print(df.head())


# -----------------------------
# 5. PREP FEATURES AND TARGET
# -----------------------------
drop_cols = [
    "member_id",
    "snapshot_date",
    "future_6m_total_cost"   # leakage - must stay out
]

X = df.drop(columns=drop_cols + ["high_cost_next_6m_flag"])
y = df["high_cost_next_6m_flag"]

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

print("\nFeature matrix shape after encoding:", X.shape)


# -----------------------------
# 6. TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# -----------------------------
# 7. LOGISTIC REGRESSION
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

y_pred_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]
y_pred_lr = (y_pred_proba_lr >= THRESHOLD).astype(int)

print("\n=== Logistic Regression ===")
print("AUC:", roc_auc_score(y_test, y_pred_proba_lr))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))
print("Classification Report:")
print(classification_report(y_test, y_pred_lr, zero_division=0))


# -----------------------------
# 8. RANDOM FOREST
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)

y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]
y_pred_rf = (y_pred_proba_rf >= THRESHOLD).astype(int)

print("\n=== Random Forest ===")
print("AUC:", roc_auc_score(y_test, y_pred_proba_rf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("Classification Report:")
print(classification_report(y_test, y_pred_rf, zero_division=0))


# -----------------------------
# 9. XGBOOST
# -----------------------------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(X_train, y_train)

y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_pred_xgb = (y_pred_proba_xgb >= THRESHOLD).astype(int)

print("\n=== XGBoost ===")
print("AUC:", roc_auc_score(y_test, y_pred_proba_xgb))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))
print("Classification Report:")
print(classification_report(y_test, y_pred_xgb, zero_division=0))


# -----------------------------
# 🔍 SHAP EXPLAINABILITY
# -----------------------------
import shap
import matplotlib.pyplot as plt

X_sample = X_test.sample(min(1000, len(X_test)), random_state=42)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_sample)

shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_bar.png", bbox_inches="tight")
plt.close()

shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", bbox_inches="tight")
plt.close()

print("Saved SHAP plots: shap_bar.png, shap_summary.png")


# -----------------------------
# 🔍 THRESHOLD ANALYSIS (XGBoost)
# -----------------------------
from sklearn.metrics import precision_recall_curve
import numpy as np
import pandas as pd

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba_xgb)

pr_df = pd.DataFrame({
    "threshold": np.append(thresholds, np.nan),
    "precision": precision,
    "recall": recall
})

print("\nThreshold tuning preview (first 20 rows):")
print(pr_df.head(20))


# -----------------------------
# 10. MODEL COMPARISON
# -----------------------------
lr_auc = roc_auc_score(y_test, y_pred_proba_lr)
rf_auc = roc_auc_score(y_test, y_pred_proba_rf)
xgb_auc = roc_auc_score(y_test, y_pred_proba_xgb)

print("\n=== Model Comparison ===")
print(f"Logistic Regression AUC: {lr_auc:.3f}")
print(f"Random Forest AUC:       {rf_auc:.3f}")
print(f"XGBoost AUC:             {xgb_auc:.3f}")

best_model = xgb_model
best_model_name = "XGBoost"


# -----------------------------
# 11. FEATURE IMPORTANCE
# -----------------------------
importances = pd.Series(best_model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(15)

print(f"\nTop 15 Feature Importances ({best_model_name}):")
print(top_features)


# -----------------------------
# 12. SCORE ALL MEMBERS
# -----------------------------
X_all = pd.get_dummies(
    df.drop(columns=drop_cols + ["high_cost_next_6m_flag"]),
    drop_first=True
)

X_all = X_all.reindex(columns=X.columns, fill_value=0)

df["risk_score_pred"] = best_model.predict_proba(X_all)[:, 1]

print(f"\nTop 20 High-Risk Members ({best_model_name}):")
print(df.sort_values("risk_score_pred", ascending=False).head(20))


# -----------------------------
# 13. TOP 10% CAPTURE
# -----------------------------
df_ranked = df.sort_values("risk_score_pred", ascending=False).copy()

top_10_pct = int(len(df_ranked) * 0.10)
df_ranked["predicted_high_risk"] = 0
df_ranked.iloc[:top_10_pct, df_ranked.columns.get_loc("predicted_high_risk")] = 1

print("\nTop 10% capture matrix:")
print(pd.crosstab(df_ranked["predicted_high_risk"], df_ranked["high_cost_next_6m_flag"]))


# -----------------------------
# 14. RISK DECILES
# -----------------------------
df_ranked["risk_decile"] = pd.qcut(
    df_ranked["risk_score_pred"],
    10,
    labels=False,
    duplicates="drop"
)

print("\nObserved high-cost rate by risk decile:")
print(df_ranked.groupby("risk_decile")["high_cost_next_6m_flag"].mean())


# -----------------------------
# 15. SAVE SCORED OUTPUT
# -----------------------------
df_ranked.to_csv("member_risk_scores.csv", index=False)
print("\nSaved scored file: member_risk_scores.csv")


#Temp Block
print([c for c in df.columns if "cost" in c.lower()])
print([c for c in df.columns if "claim" in c.lower()])
print([c for c in df.columns if "visit" in c.lower()])
print([c for c in df.columns if "dx" in c.lower()])


