
from __future__ import annotations
 
import numpy as np
import pandas as pd
 
# ============================================================================
# 1. FEATURE CONTRACT
# ============================================================================
 
ID_COLUMNS = ["booking_id", "worker_id"]      # metadata only, never model input
GROUP_COLUMN = "booking_id"                    # groups candidates for split/ranking, not a feature
TARGET_COLUMN = "successful_completion"
 
# Candidate-varying + booking-context raw features that are safe to use.
RAW_NUMERIC_FEATURES = [
    "quantity", "scheduled_hour", "day_of_week",   # booking context (constant per booking - pointwise calibration only)
    "experience_years", "certifications_count", "has_certifications",
    "distance_km", "worker_skill_price", "platform_tenure_days",
]
CATEGORICAL_FEATURES = ["worker_skill_price_type"]
 
# EXCLUDED, and why (production-consistency decision, not an oversight):
#   is_verified       -> hard eligibility constraint; backend only ever sends
#                         verified workers, so this is constant (=1) at real
#                         inference time even though it varies in training data.
#   worker_category    -> hard eligibility constraint; backend's candidate
#                         query already filters on this; constant per booking.
#   service_category    -> same as worker_category; the filter criterion itself.
#   skill_name          -> used upstream to filter candidates and look up
#                         worker_skill_price/type, but constant within a
#                         booking's candidate set and confirmed to carry no
#                         real target signal (by design). Not a model input.
EXCLUDED_FEATURES = ["is_verified", "worker_category", "service_category", "skill_name"]
 
# Within-booking relative (z-scored) counterpart of each candidate-varying
# raw feature - "closer/more experienced/cheaper THAN THE OTHER CANDIDATES
# FOR THIS BOOKING", not just an absolute number. Reproducible identically
# at inference because one API request = the full candidate list for one
# booking = exactly the group needed to compute this.
RELATIVE_SOURCE_COLUMNS = ["distance_km", "experience_years", "certifications_count", "worker_skill_price", "platform_tenure_days"]
RELATIVE_FEATURES = [f"{c}_rel" for c in RELATIVE_SOURCE_COLUMNS]
 
NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + RELATIVE_FEATURES
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
 
 
def validate_schema(df: pd.DataFrame, require_target: bool = True) -> None:
    """Fail loudly and specifically on any schema mismatch - training data
    and inference requests must conform to the exact same raw contract."""
    raw_cols = RAW_NUMERIC_FEATURES + CATEGORICAL_FEATURES
    required = raw_cols + [GROUP_COLUMN] + ([TARGET_COLUMN] if require_target else [])
 
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Schema validation failed. Missing column(s): {missing}")
 
    nulls = df[raw_cols].isnull().sum()
    bad = nulls[nulls > 0]
    if len(bad) > 0:
        raise ValueError(f"Schema validation failed. Nulls found in:\n{bad}")
 
    if require_target:
        bad_targets = set(df[TARGET_COLUMN].unique()) - {0, 1}
        if bad_targets:
            raise ValueError(f"Target column contains values outside {{0,1}}: {bad_targets}")
 
 
# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
 
def _zscore_within_group(s: pd.Series) -> pd.Series:
    std = s.std()
    if not std or std < 1e-9:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std
 
 
def add_relative_features(df: pd.DataFrame, group_col: str = GROUP_COLUMN) -> pd.DataFrame:
    """Adds one *_rel column per RELATIVE_SOURCE_COLUMNS: the z-score of
    that value against the other rows sharing the same group_col value.
    At training time, group_col has 3,000 groups (all bookings). At
    inference time, one API call = one group (that booking's candidates)."""
    out = df.copy()
    for col in RELATIVE_SOURCE_COLUMNS:
        out[f"{col}_rel"] = out.groupby(group_col)[col].transform(_zscore_within_group)
    return out
 
 
# ============================================================================
# 3. RANKING METRICS
# ============================================================================
 
def _dcg_at_k(relevances: np.ndarray, k: int) -> float:
    relevances = relevances[:k]
    if len(relevances) == 0:
        return 0.0
    gains = (2.0 ** relevances) - 1.0
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return float(np.sum(gains / discounts))
 
 
def _ndcg_at_k(sorted_relevances: np.ndarray, k: int) -> float | None:
    dcg = _dcg_at_k(sorted_relevances, k)
    ideal = _dcg_at_k(np.sort(sorted_relevances)[::-1], k)
    return None if ideal == 0.0 else dcg / ideal
 
 
def evaluate_ranking(df: pd.DataFrame, score_col: str, label_col: str, group_col: str, ks=(3, 5)) -> dict:
    """Groups by group_col (booking) and computes NDCG@k, Hit@1, MRR -
    "does the model rank the more suitable candidates above the less
    suitable ones for a given booking?" - the real objective, as opposed
    to row-level classification metrics."""
    ndcg_values = {k: [] for k in ks}
    excluded = 0
    hits, rr = [], []
 
    for _, g in df.groupby(group_col):
        g_sorted = g.sort_values(score_col, ascending=False)
        y = g_sorted[label_col].to_numpy()
        if y.sum() == 0:
            excluded += 1
        for k in ks:
            v = _ndcg_at_k(y, k)
            if v is not None:
                ndcg_values[k].append(v)
        hits.append(1.0 if y[0] == 1 else 0.0)
        pos = np.where(y == 1)[0]
        rr.append(1.0 / (pos[0] + 1) if len(pos) else 0.0)
 
    result = {f"ndcg@{k}": (float(np.mean(v)) if v else None) for k, v in ndcg_values.items()}
    result.update({
        "ndcg_groups_excluded_no_positive": excluded,
        "hit_at_1": float(np.mean(hits)),
        "mean_reciprocal_rank": float(np.mean(rr)),
    })
    return result
 