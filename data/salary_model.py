"""
Predicting annual_salary_usd from the Stack Overflow 2025 Developer Survey subset.

Run with:  python salary_model.py

Structure:
  1. Load data
  2. Clean / decide what to do with each column (documented inline)
  3. Feature engineering
  4. Train/test split + preprocessing pipeline
  5. Compare a few models with cross-validation
  6. Fit the best one, evaluate on the held-out test set
  7. Save predictions + metrics
"""

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DATA_PATH = "data/survey.csv"

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# 2. Cleaning decisions, column by column
# ---------------------------------------------------------------------------
# ResponseId: just a row identifier, no predictive value -> drop
df = df.drop(columns=["ResponseId"])

# --- Target column: annual_salary_usd -----------------------------------
# 84 rows have no salary at all -> can't use them for training, drop.
# The raw distribution is extremely skewed and contains values that are not
# plausible ANNUAL salaries for an employed professional developer:
#   - 124 rows report < $1,000/year, some as low as $1-$10.
#     These look like unit/typo errors (e.g. someone entered a monthly figure,
#     or a currency-conversion artifact for a low-value local currency)
#     rather than real full-time comp.
#   - A handful of rows (8) report > $1,000,000, up to $6.89M. These may be
#     genuine (equity-heavy exec comp) but they are extreme leverage points
#     with almost no similar neighbors to learn from, so a model will not be
#     able to predict them reliably regardless of features.
# Decision: drop rows with salary < $2,000 or > $1,000,000. This removes
# ~5% of rows and is a judgement call, not a scientifically derived cutoff -
# chosen by eyeballing the quantiles and the raw values.
before = len(df)
df = df.dropna(subset=["annual_salary_usd"])
df = df[(df["annual_salary_usd"] >= 2000) & (df["annual_salary_usd"] <= 1_000_000)]
print(f"Dropped {before - len(df)} rows for missing/implausible salary -> {len(df)} rows remain")

# --- Age -------------------------------------------------------------------
# Reported as text ranges (e.g. "25-34 years old"). Treat as ordinal and map
# each bucket to its midpoint so the model gets a single numeric feature
# instead of exploding it into 7 dummy columns for what is really an ordered
# quantity. "Prefer not to say" (6 rows) is treated as missing and imputed
# later along with genuine NaNs.
AGE_MAP = {
    "18-24 years old": 21,
    "25-34 years old": 29.5,
    "35-44 years old": 39.5,
    "45-54 years old": 49.5,
    "55-64 years old": 59.5,
    "65 years or older": 68,
}
df["age_numeric"] = df["Age"].map(AGE_MAP)  # unmapped values (Prefer not to say) -> NaN

# --- WorkExp / YearsCode ----------------------------------------------------
# Already numeric (years). Some missingness (WorkExp ~1%, YearsCode <1%),
# imputed with the median inside the pipeline. Noted: ~4% of rows have
# WorkExp > YearsCode (e.g. people who worked before they coded professionally,
# or manager time counted differently) - not treated as an error, left as-is.

# --- EdLevel, Employment, DevType, OrgSize, ICorPM, RemoteWork, Industry ---
# All free-choice categorical columns from the survey. Missing values here
# don't mean "zero" the way a missing language list might - they mean the
# respondent skipped or the question didn't apply, so they're filled with an
# explicit "Unknown" category rather than the column mode. This lets the
# model treat "didn't answer" as its own signal instead of pretending it's
# more common than it is.
CATEGORICAL_UNKNOWN_FILL = ["EdLevel", "OrgSize", "ICorPM", "RemoteWork", "Industry"]
for col in CATEGORICAL_UNKNOWN_FILL:
    df[col] = df[col].fillna("Unknown")

# Employment has no missing values, keep as-is.

# --- Country -----------------------------------------------------------
# 130 distinct countries, most with a handful of respondents. One-hot
# encoding all of them would create mostly-empty columns that easily overfit
# in a 5,000-row dataset. Keep the 25 most frequent countries (~81% of rows)
# and bucket everything else into "Other" - salary differences are large and
# real between countries (cost of living, currency), so this is worth keeping
# as a feature, just not at full resolution.
TOP_N_COUNTRIES = 25
top_countries = df["Country"].value_counts().nlargest(TOP_N_COUNTRIES).index
df["country_grouped"] = np.where(df["Country"].isin(top_countries), df["Country"], "Other")

# --- DevType -------------------------------------------------------------
# 32 categories but a long tail; group anything under 1% of rows into "Other"
# for the same reason as Country.
devtype_counts = df["DevType"].value_counts(normalize=True)
rare_devtypes = devtype_counts[devtype_counts < 0.01].index
df["devtype_grouped"] = df["DevType"].where(~df["DevType"].isin(rare_devtypes), "Other")

# --- Currency ------------------------------------------------------------
# Dropped. Two reasons: (1) formatting is inconsistent - some values are
# tab-separated ("AUD\tAustralian dollar") and some space-separated
# ("EUR European Euro"), a sign this column wasn't cleaned before export;
# (2) since annual_salary_usd is already converted to USD, currency mostly
# just re-encodes Country (same information, messier). Country is kept
# instead as the cleaner version of that signal.

# --- LanguageHaveWorkedWith / DatabaseHaveWorkedWith ----------------------
# Multi-select fields, semicolon-separated. Missing here plausibly means
# "reported none" rather than "unknown", so it's filled with an empty string
# before splitting (-> 0 items) instead of "Unknown".
# Turned into: (a) a count feature (breadth of stack tends to correlate with
# seniority/pay) and (b) a few individual language flags for languages that
# show up often and plausibly carry their own salary signal (e.g. Python/Go/
# Rust vs. more common general-purpose languages).
def split_multiselect(series):
    return series.fillna("").apply(lambda s: [x for x in s.split(";") if x])

df["languages_list"] = split_multiselect(df["LanguageHaveWorkedWith"])
df["databases_list"] = split_multiselect(df["DatabaseHaveWorkedWith"])

df["num_languages"] = df["languages_list"].apply(len)
df["num_databases"] = df["databases_list"].apply(len)

FLAG_LANGUAGES = ["Python", "JavaScript", "TypeScript", "SQL", "Java", "C#", "Go", "Rust"]
for lang in FLAG_LANGUAGES:
    df[f"lang_{lang}"] = df["languages_list"].apply(lambda l, lang=lang: int(lang in l))

# ---------------------------------------------------------------------------
# 3. Target transform
# ---------------------------------------------------------------------------
# Salary is heavily right-skewed even after outlier removal (median ~$78k,
# but a long tail up to $1M). Modeling log(salary) instead of raw salary
# keeps the loss from being dominated by the handful of highest earners and
# is standard practice for income data. Predictions are converted back with
# expm1 before computing dollar-denominated error metrics.
df["log_salary"] = np.log1p(df["annual_salary_usd"])

# ---------------------------------------------------------------------------
# 4. Feature set + train/test split
# ---------------------------------------------------------------------------
numeric_features = [
    "age_numeric", "WorkExp", "YearsCode", "num_languages", "num_databases",
] + [f"lang_{lang}" for lang in FLAG_LANGUAGES]

categorical_features = [
    "EdLevel", "Employment", "devtype_grouped", "OrgSize", "ICorPM",
    "RemoteWork", "Industry", "country_grouped",
]

feature_cols = numeric_features + categorical_features
X = df[feature_cols]
y_log = df["log_salary"]
y_dollars = df["annual_salary_usd"]


#turns raw columns into model ready numbers
X_train, X_test, y_train_log, y_test_log, y_train_dollars, y_test_dollars = train_test_split(
    X, y_log, y_dollars, test_size=0.2, random_state=RANDOM_STATE
)
print(f"Train: {len(X_train)} rows, Test: {len(X_test)} rows")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_features),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features),
    ]
)

# ---------------------------------------------------------------------------
# 5. Compare models with 5-fold cross-validation on the training set
# ---------------------------------------------------------------------------
models = {
    "baseline_median": DummyRegressor(strategy="median"),
    "ridge_regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "random_forest": RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=3,
        random_state=RANDOM_STATE, n_jobs=-1,
    ),
    "gradient_boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
}

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    # Predict in log-space via CV, then convert back to dollars to score
    # in units that are actually interpretable (MAE in dollars).
    preds_log = cross_val_predict(pipe, X_train, y_train_log, cv=cv, n_jobs=-1)
    preds_dollars = np.expm1(preds_log)
    mae = mean_absolute_error(y_train_dollars, preds_dollars)
    rmse = np.sqrt(mean_squared_error(y_train_dollars, preds_dollars))
    r2 = r2_score(y_train_dollars, preds_dollars)
    cv_results[name] = {"cv_mae": mae, "cv_rmse": rmse, "cv_r2": r2}
    print(f"{name:20s}  CV MAE=${mae:,.0f}  CV RMSE=${rmse:,.0f}  CV R2={r2:.3f}")

# ---------------------------------------------------------------------------
# 6. Fit the best model on the full training set, evaluate on the held-out test set
# ---------------------------------------------------------------------------
best_name = min(cv_results, key=lambda k: cv_results[k]["cv_mae"])
print(f"\nBest model by CV MAE: {best_name}")

final_pipe = Pipeline([("prep", preprocessor), ("model", models[best_name])])
final_pipe.fit(X_train, y_train_log)

test_preds_log = final_pipe.predict(X_test)
test_preds_dollars = np.expm1(test_preds_log)

test_mae = mean_absolute_error(y_test_dollars, test_preds_dollars)
test_rmse = np.sqrt(mean_squared_error(y_test_dollars, test_preds_dollars))
test_r2 = r2_score(y_test_dollars, test_preds_dollars)
test_medae_pct = np.median(np.abs(test_preds_dollars - y_test_dollars) / y_test_dollars) * 100

print(f"\nHeld-out test set ({best_name}):")
print(f"  MAE  = ${test_mae:,.0f}")
print(f"  RMSE = ${test_rmse:,.0f}")
print(f"  R2   = {test_r2:.3f}")
print(f"  Median absolute % error = {test_medae_pct:.1f}%")

# Baseline for comparison: always predict the training median salary
naive_pred = np.full_like(y_test_dollars, y_train_dollars.median(), dtype=float)
naive_mae = mean_absolute_error(y_test_dollars, naive_pred)
print(f"  (naive median-guess MAE on the same test set = ${naive_mae:,.0f})")

# ---------------------------------------------------------------------------
# 7. Feature importance (for the tree-based winner) + save outputs
# ---------------------------------------------------------------------------
feature_names = numeric_features + list(
    final_pipe.named_steps["prep"].named_transformers_["cat"]
    .named_steps["onehot"].get_feature_names_out(categorical_features)
)

importances = None
model_obj = final_pipe.named_steps["model"]
if hasattr(model_obj, "feature_importances_"):
    importances = pd.Series(model_obj.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\nTop 15 feature importances:")
    print(importances.head(15))
elif hasattr(model_obj, "coef_"):
    importances = pd.Series(model_obj.coef_, index=feature_names).sort_values(key=abs, ascending=False)
    print("\nTop 15 coefficients (log-salary scale):")
    print(importances.head(15))

# Save predictions on the test set
out = X_test.copy()
out["actual_salary_usd"] = y_test_dollars.values
out["predicted_salary_usd"] = test_preds_dollars
out["abs_error"] = (out["actual_salary_usd"] - out["predicted_salary_usd"]).abs()
out.to_csv("predictions.csv", index=False)

metrics = {
    "n_rows_used": len(df),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "cv_results_on_train": cv_results,
    "best_model": best_name,
    "test_mae_usd": test_mae,
    "test_rmse_usd": test_rmse,
    "test_r2": test_r2,
    "test_median_abs_pct_error": test_medae_pct,
    "naive_median_baseline_mae_usd": naive_mae,
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved predictions.csv and metrics.json")
