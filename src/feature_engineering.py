"""
feature_engineering.py
-----------------------
Transforms raw account-month data into analysis-ready features:
  - DPD bucket classification
  - Roll rate transition matrix
  - Charge-off and recovery timing
  - Cohort-level loss build schedule
"""

import pandas as pd
import numpy as np


# ── DPD Bucketing ─────────────────────────────────────────────────────────────

DPD_ORDER = ["current", "dpd_30", "dpd_60", "dpd_90", "dpd_120", "charged_off", "paid_off"]
DPD_LABELS = {
    "current":     "Current",
    "dpd_30":      "30 DPD",
    "dpd_60":      "60 DPD",
    "dpd_90":      "90 DPD",
    "dpd_120":     "120 DPD",
    "charged_off": "Charged Off",
    "paid_off":    "Paid Off",
}


def label_dpd_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Add a display-friendly DPD bucket label column."""
    df = df.copy()
    df["dpd_bucket"] = df["dpd_state"].map(DPD_LABELS)
    df["dpd_bucket"] = pd.Categorical(
        df["dpd_bucket"],
        categories=[DPD_LABELS[s] for s in DPD_ORDER],
        ordered=True,
    )
    return df


# ── Roll Rate Matrix ──────────────────────────────────────────────────────────

def compute_roll_rates(df: pd.DataFrame, vintage: str = None) -> pd.DataFrame:
    """
    Compute month-over-month roll rate transition matrix.
    Optionally filter to a single vintage.

    Returns a DataFrame shaped (from_state x to_state) with transition rates.
    """
    work = df.copy()
    if vintage:
        work = work[work["vintage"] == vintage]

    work = work.sort_values(["account_id", "months_on_book"])
    work["next_state"] = work.groupby("account_id")["dpd_state"].shift(-1)
    work = work.dropna(subset=["next_state"])

    # Filter out terminal states as origin (no meaningful transitions)
    active_states = ["current", "dpd_30", "dpd_60", "dpd_90", "dpd_120"]
    work = work[work["dpd_state"].isin(active_states)]

    counts = work.groupby(["dpd_state", "next_state"]).size().unstack(fill_value=0)
    rates = counts.div(counts.sum(axis=1), axis=0)

    # Reorder rows/cols to match DPD_ORDER
    all_states = [s for s in DPD_ORDER if s in rates.index or s in rates.columns]
    rates = rates.reindex(index=all_states, columns=all_states, fill_value=0)

    return rates.round(4)


# ── Cohort Loss Build ─────────────────────────────────────────────────────────

def compute_loss_build(vintage_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a loss build schedule per vintage:
    - Monthly gross loss
    - Monthly recovery
    - Monthly net loss
    - Cumulative net loss rate at each MOB
    """
    cols = [
        "vintage", "months_on_book",
        "gross_loss_amt", "recovery_amt",
        "cumulative_net_loss", "net_loss_rate",
        "cohort_orig_balance", "cohort_size",
    ]
    return vintage_summary[cols].copy()


# ── Period-over-Period Comparison ─────────────────────────────────────────────

def compare_vintages_at_mob(vintage_summary: pd.DataFrame, mob: int, metric: str = "net_loss_rate") -> pd.DataFrame:
    """
    Snapshot all vintages at a specific months-on-book.
    Useful for benchmarking cohort performance at a fixed loan age.

    metric options: net_loss_rate, dpd_30_rate, dpd_60_rate, dpd_90_rate
    """
    snap = vintage_summary[vintage_summary["months_on_book"] == mob][
        ["vintage", metric, "cohort_size", "cohort_orig_balance"]
    ].copy()
    snap = snap.sort_values("vintage")
    portfolio_avg = snap[metric].mean()
    snap["vs_portfolio_avg"] = snap[metric] - portfolio_avg
    return snap


# ── Summary Stats ─────────────────────────────────────────────────────────────

def portfolio_summary_stats(df: pd.DataFrame, vintage_summary: pd.DataFrame) -> dict:
    """
    Returns top-line portfolio metrics for the executive summary section
    of the notebook.
    """
    total_accounts = df["account_id"].nunique()
    total_orig_balance = df[df["months_on_book"] == 1]["origination_balance"].sum()
    pct_ever_delinquent = (
        df[df["is_30_dpd"] == 1]["account_id"].nunique() / total_accounts
    )
    pct_charged_off = (
        df[df["is_charged_off"] == 1]["account_id"].nunique() / total_accounts
    )
    avg_mob_to_co = (
        df[df["dpd_state"] == "charged_off"]
        .groupby("account_id")["months_on_book"]
        .min()
        .mean()
    )
    max_mob_net_loss_rate = vintage_summary.groupby("vintage")["net_loss_rate"].max()

    return {
        "total_accounts": total_accounts,
        "total_orig_balance_mm": round(total_orig_balance / 1e6, 2),
        "pct_ever_delinquent": round(pct_ever_delinquent * 100, 1),
        "pct_charged_off": round(pct_charged_off * 100, 1),
        "avg_months_to_charge_off": round(avg_mob_to_co, 1),
        "net_loss_rate_by_vintage": max_mob_net_loss_rate.round(4).to_dict(),
    }
