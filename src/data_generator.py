"""
data_generator.py
-----------------
Generates a synthetic consumer credit / BNPL portfolio dataset
for vintage delinquency and net loss rate analysis.

Each row represents one account-month observation.
Cohorts are defined by origination quarter (vintage).

Usage:
    python src/data_generator.py
    # Writes data/raw/portfolio_account_months.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path
import random

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Parameters (edit these to stress-test different scenarios) ────────────────
CONFIG = {
    "n_accounts": 8_000,
    "vintages": [
        "2022-Q1", "2022-Q2", "2022-Q3", "2022-Q4",
        "2023-Q1", "2023-Q2", "2023-Q3", "2023-Q4",
    ],
    "max_months_on_book": 24,
    "risk_tiers": {
        "prime":      {"weight": 0.45, "base_default_rate": 0.02},
        "near_prime": {"weight": 0.35, "base_default_rate": 0.06},
        "subprime":   {"weight": 0.20, "base_default_rate": 0.14},
    },
    "avg_origination_balance": 2_400,
    "balance_std": 800,
    "min_balance": 300,
    "max_balance": 8_000,
    "charge_off_dpd_threshold": 120,   # accounts at 120+ DPD are charged off next month
    "recovery_rate": 0.15,             # 15% recovered 6 months post charge-off
    # Macro stress: later vintages carry slightly higher baseline delinquency
    # (simulates 2023 tightening credit environment)
    "macro_stress_by_vintage": {
        "2022-Q1": 0.00,
        "2022-Q2": 0.00,
        "2022-Q3": 0.01,
        "2022-Q4": 0.01,
        "2023-Q1": 0.02,
        "2023-Q2": 0.03,
        "2023-Q3": 0.04,
        "2023-Q4": 0.04,
    },
}

# ── DPD State Machine ─────────────────────────────────────────────────────────
# States: current, dpd_30, dpd_60, dpd_90, dpd_120, charged_off, paid_off
# Transition probabilities vary by risk tier and months on book

def get_transition_matrix(risk_tier: str, mob: int, macro_stress: float) -> dict:
    """
    Returns transition probabilities from each DPD state.
    Probability of moving from 'current' into first delinquency
    increases slightly with months on book (seasoning effect), then plateaus.
    """
    base = CONFIG["risk_tiers"][risk_tier]["base_default_rate"]
    seasoning = min(1.0 + (mob / 24) * 0.5, 1.5)  # ramp up over first 24 months
    p_miss = min((base + macro_stress) * seasoning, 0.40)

    return {
        "current": {
            "current":     1 - p_miss - 0.005,
            "dpd_30":      p_miss,
            "paid_off":    0.005,
        },
        "dpd_30": {
            "current":     0.55,   # cure rate from 30 DPD
            "dpd_30":      0.10,
            "dpd_60":      0.35,
        },
        "dpd_60": {
            "current":     0.20,   # cure rate from 60 DPD
            "dpd_30":      0.10,
            "dpd_60":      0.10,
            "dpd_90":      0.60,
        },
        "dpd_90": {
            "current":     0.05,
            "dpd_60":      0.05,
            "dpd_90":      0.20,
            "dpd_120":     0.70,
        },
        "dpd_120": {
            "dpd_120":     0.10,
            "charged_off": 0.90,
        },
        "charged_off": {
            "charged_off": 1.0,
        },
        "paid_off": {
            "paid_off": 1.0,
        },
    }


def next_state(current_state: str, risk_tier: str, mob: int, macro_stress: float) -> str:
    matrix = get_transition_matrix(risk_tier, mob, macro_stress)
    transitions = matrix[current_state]
    states = list(transitions.keys())
    probs = list(transitions.values())
    # normalize to ensure they sum to 1.0 (floating point safety)
    total = sum(probs)
    probs = [p / total for p in probs]
    return np.random.choice(states, p=probs)


# ── Account Generation ────────────────────────────────────────────────────────

def generate_accounts(config: dict) -> pd.DataFrame:
    """Generate one row per account with static attributes."""
    n = config["n_accounts"]
    vintages = config["vintages"]

    risk_tiers = list(config["risk_tiers"].keys())
    tier_weights = [config["risk_tiers"][t]["weight"] for t in risk_tiers]

    accounts = pd.DataFrame({
        "account_id": [f"ACC{str(i).zfill(6)}" for i in range(1, n + 1)],
        "vintage": np.random.choice(vintages, size=n),
        "risk_tier": np.random.choice(risk_tiers, size=n, p=tier_weights),
        "origination_balance": np.clip(
            np.random.normal(
                config["avg_origination_balance"],
                config["balance_std"],
                size=n
            ),
            config["min_balance"],
            config["max_balance"]
        ).round(2),
    })

    # Assign origination date (random month within the vintage quarter)
    def vintage_to_date(vintage: str) -> pd.Timestamp:
        year, q = vintage.split("-Q")
        q = int(q)
        start_month = (q - 1) * 3 + 1
        month = random.randint(start_month, start_month + 2)
        day = random.randint(1, 28)
        return pd.Timestamp(year=int(year), month=month, day=day)

    accounts["origination_date"] = accounts["vintage"].apply(vintage_to_date)
    return accounts


# ── Monthly State Simulation ──────────────────────────────────────────────────

def simulate_portfolio(accounts: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Walk each account through the DPD state machine for up to max_months_on_book.
    Returns a long-format DataFrame with one row per account per month.
    """
    records = []
    max_mob = config["max_months_on_book"]

    for _, acct in accounts.iterrows():
        state = "current"
        orig_balance = acct["origination_balance"]
        vintage = acct["vintage"]
        macro_stress = config["macro_stress_by_vintage"][vintage]
        risk_tier = acct["risk_tier"]
        charged_off_month = None
        recovery_applied = False

        for mob in range(1, max_mob + 1):
            obs_date = acct["origination_date"] + pd.DateOffset(months=mob)

            # Balance decays as account pays down (simplistic linear paydown)
            if state not in ("charged_off", "paid_off"):
                remaining_balance = orig_balance * max(0, 1 - (mob / (max_mob * 1.5)))
            else:
                remaining_balance = 0.0

            # Recovery logic: 6 months post charge-off
            recovery_amount = 0.0
            if state == "charged_off":
                if charged_off_month and (mob - charged_off_month) == 6 and not recovery_applied:
                    recovery_amount = orig_balance * config["recovery_rate"]
                    recovery_applied = True

            records.append({
                "account_id":          acct["account_id"],
                "vintage":             vintage,
                "risk_tier":           risk_tier,
                "origination_balance": orig_balance,
                "origination_date":    acct["origination_date"],
                "months_on_book":      mob,
                "observation_date":    obs_date,
                "dpd_state":           state,
                "remaining_balance":   round(remaining_balance, 2),
                "recovery_amount":     round(recovery_amount, 2),
            })

            # Transition to next state
            new_state = next_state(state, risk_tier, mob, macro_stress)
            if new_state == "charged_off" and state != "charged_off":
                charged_off_month = mob
            state = new_state

            if state in ("paid_off",):
                break

    df = pd.DataFrame(records)

    # ── Derived fields ─────────────────────────────────────────────────────────
    df["is_30_dpd"]      = df["dpd_state"].isin(["dpd_30", "dpd_60", "dpd_90", "dpd_120"]).astype(int)
    df["is_60_dpd"]      = df["dpd_state"].isin(["dpd_60", "dpd_90", "dpd_120"]).astype(int)
    df["is_90_dpd"]      = df["dpd_state"].isin(["dpd_90", "dpd_120"]).astype(int)
    df["is_charged_off"] = (df["dpd_state"] == "charged_off").astype(int)
    df["gross_loss"]     = np.where(
        (df["dpd_state"] == "charged_off") & (df["months_on_book"] == df.groupby("account_id")["months_on_book"].transform(
            lambda x: x[df.loc[x.index, "dpd_state"] == "charged_off"].min() if (df.loc[x.index, "dpd_state"] == "charged_off").any() else np.nan
        )),
        df["origination_balance"],
        0.0
    )

    return df


# ── Vintage Aggregation ───────────────────────────────────────────────────────

def build_vintage_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to vintage x months_on_book level.
    Computes DPD rates and cumulative net loss rate.
    """
    # Cohort size (active accounts at MOB=1 per vintage)
    cohort_sizes = (
        df[df["months_on_book"] == 1]
        .groupby("vintage")["account_id"]
        .nunique()
        .rename("cohort_size")
    )

    cohort_balances = (
        df[df["months_on_book"] == 1]
        .groupby("vintage")["origination_balance"]
        .sum()
        .rename("cohort_orig_balance")
    )

    # Monthly snapshot per vintage x MOB
    monthly = (
        df.groupby(["vintage", "months_on_book"])
        .agg(
            active_accounts=("account_id", "nunique"),
            dpd_30_count=("is_30_dpd", "sum"),
            dpd_60_count=("is_60_dpd", "sum"),
            dpd_90_count=("is_90_dpd", "sum"),
            charged_off_count=("is_charged_off", "sum"),
            gross_loss_amt=("origination_balance", lambda x: x[df.loc[x.index, "dpd_state"] == "charged_off"].sum()),
            recovery_amt=("recovery_amount", "sum"),
        )
        .reset_index()
    )

    monthly = monthly.merge(cohort_sizes, on="vintage")
    monthly = monthly.merge(cohort_balances, on="vintage")

    # DPD rates (% of cohort accounts)
    monthly["dpd_30_rate"] = monthly["dpd_30_count"] / monthly["cohort_size"]
    monthly["dpd_60_rate"] = monthly["dpd_60_count"] / monthly["cohort_size"]
    monthly["dpd_90_rate"] = monthly["dpd_90_count"] / monthly["cohort_size"]

    # Cumulative net loss rate (% of original cohort balance)
    monthly = monthly.sort_values(["vintage", "months_on_book"])
    monthly["cumulative_gross_loss"] = monthly.groupby("vintage")["gross_loss_amt"].cumsum()
    monthly["cumulative_recovery"]   = monthly.groupby("vintage")["recovery_amt"].cumsum()
    monthly["cumulative_net_loss"]   = monthly["cumulative_gross_loss"] - monthly["cumulative_recovery"]
    monthly["net_loss_rate"]         = monthly["cumulative_net_loss"] / monthly["cohort_orig_balance"]

    return monthly


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Generating accounts...")
    accounts = generate_accounts(CONFIG)
    print(f"  {len(accounts):,} accounts across {len(CONFIG['vintages'])} vintages")

    print("Simulating monthly states...")
    df = simulate_portfolio(accounts, CONFIG)
    print(f"  {len(df):,} account-month observations generated")

    print("Building vintage summary...")
    vintage_summary = build_vintage_summary(df)

    # Save outputs
    raw_path = Path("data/raw/portfolio_account_months.parquet")
    summary_path = Path("data/processed/vintage_summary.parquet")

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(raw_path, index=False)
    vintage_summary.to_parquet(summary_path, index=False)

    print(f"\nSaved:")
    print(f"  {raw_path}  ({len(df):,} rows)")
    print(f"  {summary_path}  ({len(vintage_summary):,} rows)")
    print("\nDone. Open notebooks/vintage_analysis.ipynb to run the full analysis.")


if __name__ == "__main__":
    main()
