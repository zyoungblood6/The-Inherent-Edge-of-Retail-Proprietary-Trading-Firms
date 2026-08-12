from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# AUDIT — FINAL COMPLETE 10:00 / 100-TICK / 8-MNQ TEST
#
# Audits:
#   1) Raw strategy full sample
#   2) Raw exact five-year split
#   3) One-account full Topstep Option 2 result
#   4) Five-account full portfolio scaling
#   5) Five-account exact five-year split
#   6) Full-sample payout/billing/MLL audit logs
#
# This script DOES NOT rerun the Monte Carlo simulation.
# It independently checks the saved outputs from:
#   topstep_option2_5x50k_10am_COMPLETE.py
# ============================================================

# -----------------------------
# INPUT FILES
# -----------------------------

NQ_FILE = "NQ_data.csv"

DAY_FILE = "mnq_10am_100tick_8con_clean_days.csv"
PATH_FILE = "mnq_10am_100tick_8con_10000_paths.npy"

RAW_SUMMARY_FILE = (
    "raw_10am_100tick_8mnq_full_and_5year_split.csv"
)

ONE_FULL_FILE = (
    "topstep_option2_10am_100tick_8con_10000_results.csv"
)

FIVE_FULL_FILE = (
    "topstep_option2_5x50k_10am_full_results.csv"
)

FIVE_FIRST_FILE = (
    "topstep_option2_5x50k_10am_first5_results.csv"
)

FIVE_SECOND_FILE = (
    "topstep_option2_5x50k_10am_second5_results.csv"
)

FIVE_SUMMARY_FILE = (
    "topstep_option2_5x50k_10am_full_and_5year_summary.csv"
)

ACCOUNT_LOG_FILE = (
    "topstep_option2_10am_100tick_8con_account_log.csv"
)

MLL_AUDIT_FILE = (
    "topstep_option2_10am_100tick_8con_mll_audit.csv"
)

BILLING_AUDIT_FILE = (
    "topstep_option2_10am_100tick_8con_billing_audit.csv"
)

# -----------------------------
# EXACT MODEL CONSTANTS
# -----------------------------

NUM_SIMULATIONS = 10_000
ACTIVE_ACCOUNTS = 5

ENTRY_TIME = "10:00:00"
CONTRACTS = 8
TICKS = 100
TICK_SIZE = 0.25
POINT_DISTANCE = TICKS * TICK_SIZE
MNQ_DOLLARS_PER_POINT = 2.0

GROSS_TP_SL = (
    POINT_DISTANCE
    * MNQ_DOLLARS_PER_POINT
    * CONTRACTS
)

MNQ_RT_COST_PER_CONTRACT = 1.22
ROUND_TURN_COST = (
    MNQ_RT_COST_PER_CONTRACT
    * CONTRACTS
)

COMBINE_COST = 85.0
PAID_RESET_COST = 95.0
BACK2FUNDED_COST = 549.0

PAYOUT_CAP = 6000.0
MIN_PAYOUT = 125.0
TRADER_SPLIT = 0.90
MIN_XFA_TRADING_DAYS = 3
MAX_BEST_DAY_SHARE = 0.40

FIRST_START = pd.Timestamp("2016-08-08")
FIRST_END = pd.Timestamp("2021-08-06")
SECOND_START = pd.Timestamp("2021-08-09")
SECOND_END = pd.Timestamp("2026-08-05")

TOL = 0.02

# -----------------------------
# REQUIRED FILE CHECK
# -----------------------------

required = [
    NQ_FILE,
    DAY_FILE,
    PATH_FILE,
    RAW_SUMMARY_FILE,
    ONE_FULL_FILE,
    FIVE_FULL_FILE,
    FIVE_FIRST_FILE,
    FIVE_SECOND_FILE,
    FIVE_SUMMARY_FILE,
    ACCOUNT_LOG_FILE,
    MLL_AUDIT_FILE,
    BILLING_AUDIT_FILE,
]

missing = [f for f in required if not Path(f).exists()]

if missing:
    print("MISSING REQUIRED FILES:")
    for f in missing:
        print("-", f)
    raise SystemExit(
        "\nPut this audit script in the same Trading Backtest "
        "folder as the COMPLETE test outputs."
    )

errors = []

def check(name, condition, detail=""):
    ok = bool(condition)
    print(
        f"{'PASS' if ok else 'FAIL'} | {name}"
        + (f" | {detail}" if detail else "")
    )
    if not ok:
        errors.append(name)

def close(a, b, tol=TOL):
    return np.allclose(
        np.asarray(a, dtype=float),
        np.asarray(b, dtype=float),
        atol=tol,
        rtol=0,
        equal_nan=True,
    )

# ============================================================
# LOAD DATA
# ============================================================

days = pd.read_csv(DAY_FILE)
days["date_dt"] = pd.to_datetime(days["date"]).dt.normalize()

paths = np.load(PATH_FILE)

raw_saved = pd.read_csv(RAW_SUMMARY_FILE)
one_full = pd.read_csv(ONE_FULL_FILE)
five_full = pd.read_csv(FIVE_FULL_FILE)
five_first = pd.read_csv(FIVE_FIRST_FILE)
five_second = pd.read_csv(FIVE_SECOND_FILE)
five_summary = pd.read_csv(FIVE_SUMMARY_FILE)

events = pd.read_csv(ACCOUNT_LOG_FILE)
mll = pd.read_csv(MLL_AUDIT_FILE)
billing = pd.read_csv(BILLING_AUDIT_FILE)

for df in (events, mll, billing):
    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

# ============================================================
# A. STRATEGY / PATH INTEGRITY
# ============================================================

print()
print("=" * 70)
print("A. STRATEGY / RANDOM PATH INTEGRITY")
print("=" * 70)

check(
    "10,000 random paths",
    paths.shape[0] == NUM_SIMULATIONS,
    f"found {paths.shape[0]:,}"
)

check(
    "Path columns equal clean trading days",
    paths.shape[1] == len(days),
    f"{paths.shape[1]:,} vs {len(days):,}"
)

check(
    "All path values are 0/1",
    np.isin(paths, [0, 1]).all()
)

check(
    "Expected clean trading-day count = 2,485",
    len(days) == 2485,
    f"found {len(days):,}"
)

check(
    "Strategy gross TP/SL = $400",
    abs(GROSS_TP_SL - 400.0) < 1e-9,
    f"${GROSS_TP_SL:,.2f}"
)

check(
    "8-MNQ round-turn cost = $9.76",
    abs(ROUND_TURN_COST - 9.76) < 1e-9,
    f"${ROUND_TURN_COST:,.2f}"
)

# Exact five-year split.
first_mask = (
    (days["date_dt"] >= FIRST_START)
    & (days["date_dt"] <= FIRST_END)
).to_numpy()

second_mask = (
    (days["date_dt"] >= SECOND_START)
    & (days["date_dt"] <= SECOND_END)
).to_numpy()

check(
    "First block has 1,246 clean days",
    int(first_mask.sum()) == 1246,
    f"found {int(first_mask.sum()):,}"
)

check(
    "Second block has 1,239 clean days",
    int(second_mask.sum()) == 1239,
    f"found {int(second_mask.sum()):,}"
)

check(
    "Five-year blocks cover every clean day exactly once",
    np.all(first_mask ^ second_mask)
    and int(first_mask.sum() + second_mask.sum()) == len(days)
)

# ============================================================
# B. RECOMPUTE RAW STRATEGY FROM PATH MATRIX
# ============================================================

print()
print("=" * 70)
print("B. RAW STRATEGY — INDEPENDENT RECOMPUTATION")
print("=" * 70)

long_pnl = days["long_pnl"].to_numpy(dtype=float)
short_pnl = days["short_pnl"].to_numpy(dtype=float)

long_result = days["long_result"].astype(str).to_numpy()
short_result = days["short_result"].astype(str).to_numpy()

def recompute_raw(mask):
    idx = np.where(mask)[0]
    p = paths[:, idx]

    lp = long_pnl[idx]
    sp = short_pnl[idx]

    chosen_pnl = np.where(
        p == 1,
        lp[None, :],
        sp[None, :]
    )

    lr = long_result[idx]
    sr = short_result[idx]

    chosen_result = np.where(
        p == 1,
        lr[None, :],
        sr[None, :]
    )

    wins = np.sum(chosen_result == "WIN")
    losses = np.sum(chosen_result == "LOSS")
    time_exits = np.sum(
        chosen_result == "TIME_EXIT"
    )

    resolved = wins + losses

    path_totals = chosen_pnl.sum(axis=1)

    return {
        "days": len(idx),
        "mean_wins": wins / NUM_SIMULATIONS,
        "mean_losses": losses / NUM_SIMULATIONS,
        "mean_time_exits":
            time_exits / NUM_SIMULATIONS,
        "tp_sl_win_rate":
            wins / resolved * 100.0,
        "mean_trade":
            chosen_pnl.mean(),
        "mean_total":
            path_totals.mean(),
        "median_total":
            np.median(path_totals),
        "total_observations":
            chosen_pnl.size,
    }

full_mask = np.ones(len(days), dtype=bool)

raw_calc = {
    "Full sample": recompute_raw(full_mask),
    "2016-08-08 to 2021-08-06":
        recompute_raw(first_mask),
    "2021-08-09 to 2026-08-05":
        recompute_raw(second_mask),
}

for label, calc in raw_calc.items():
    row = raw_saved[
        raw_saved["Period"] == label
    ]

    check(
        f"{label}: saved raw row exists",
        len(row) == 1
    )

    if len(row) != 1:
        continue

    row = row.iloc[0]

    mappings = [
        ("Trading days per path", "days"),
        ("Mean wins per path", "mean_wins"),
        ("Mean losses per path", "mean_losses"),
        ("Mean time exits per path", "mean_time_exits"),
        ("TP/SL win rate %", "tp_sl_win_rate"),
        ("Mean raw P&L per trade", "mean_trade"),
        ("Mean raw total P&L per path", "mean_total"),
        ("Median raw total P&L per path", "median_total"),
        (
            "Total randomized trade observations",
            "total_observations"
        ),
    ]

    for saved_col, calc_key in mappings:
        check(
            f"{label}: {saved_col}",
            close(row[saved_col], calc[calc_key]),
            f"saved={row[saved_col]} "
            f"recalc={calc[calc_key]}"
        )

check(
    "Full raw TP/SL win rate is ~50%",
    abs(
        raw_calc["Full sample"]["tp_sl_win_rate"]
        - 50.0
    ) < 0.1,
    f"{raw_calc['Full sample']['tp_sl_win_rate']:.4f}%"
)

check(
    "Full raw mean P&L/trade is near zero",
    abs(
        raw_calc["Full sample"]["mean_trade"]
    ) < 1.0,
    f"${raw_calc['Full sample']['mean_trade']:.4f}"
)

check(
    "First-block mean time exits/path = 196",
    close(
        raw_calc[
            "2016-08-08 to 2021-08-06"
        ]["mean_time_exits"],
        196.0
    )
)

check(
    "Second-block mean time exits/path = 0",
    close(
        raw_calc[
            "2021-08-09 to 2026-08-05"
        ]["mean_time_exits"],
        0.0
    )
)

# ============================================================
# C. FIVE-ACCOUNT SCALING — FULL SAMPLE
# ============================================================

print()
print("=" * 70)
print("C. FIVE-ACCOUNT FULL-SAMPLE SCALING")
print("=" * 70)

check(
    "Full one-account and five-account rows = 10,000",
    len(one_full) == NUM_SIMULATIONS
    and len(five_full) == NUM_SIMULATIONS
)

# Sort by simulation for exact row alignment.
one_full = one_full.sort_values(
    "simulation"
).reset_index(drop=True)

five_full = five_full.sort_values(
    "simulation"
).reset_index(drop=True)

check(
    "Simulation IDs align",
    np.array_equal(
        one_full["simulation"].to_numpy(),
        five_full["simulation"].to_numpy()
    )
)

scalable_cols = [
    "combine_subscriptions_purchased",
    "monthly_rebills",
    "paid_resets",
    "reset_credits_earned",
    "reset_credits_used",
    "reset_credits_expired",
    "reset_credits_remaining",
    "combine_passes",
    "xfa_accounts_earned",
    "back2funded_reactivations",
    "xfa_failures",
    "accounts_blown",
    "payouts",
    "trader_payouts",
    "account_costs",
    "actual_net_pnl",
]

for c in scalable_cols:
    if c in one_full.columns and c in five_full.columns:
        check(
            f"Full sample 5x scaling: {c}",
            close(
                five_full[c],
                one_full[c] * ACTIVE_ACCOUNTS
            )
        )

check(
    "Five synchronized accounts do not change profitable %",
    close(
        (five_full["actual_net_pnl"] > 0).mean(),
        (one_full["actual_net_pnl"] > 0).mean()
    )
)

# ============================================================
# D. RESULT-LEVEL TOPSTEP COST / P&L RECONCILIATION
# ============================================================

print()
print("=" * 70)
print("D. TOPSTEP RESULT-LEVEL RECONCILIATION")
print("=" * 70)

def result_level_audit(df, multiplier, label):
    expected_cost = (
        df["combine_subscriptions_purchased"]
        * COMBINE_COST
        + df["monthly_rebills"]
        * COMBINE_COST
        + df["paid_resets"]
        * PAID_RESET_COST
        + df["back2funded_reactivations"]
        * BACK2FUNDED_COST
    )

    # For 5x result files, all counts are already multiplied by
    # five, so no extra multiplier belongs in expected_cost.
    check(
        f"{label}: account costs reconcile",
        close(
            df["account_costs"],
            expected_cost
        )
    )

    check(
        f"{label}: actual net P&L = payouts - costs",
        close(
            df["actual_net_pnl"],
            df["trader_payouts"]
            - df["account_costs"]
        )
    )

    check(
        f"{label}: Back2Funded always OFF",
        (df["back2funded_reactivations"] == 0).all()
    )

    check(
        f"{label}: no paid resets",
        (df["paid_resets"] == 0).all()
    )

    check(
        f"{label}: rebills equal reset credits earned",
        np.array_equal(
            df["monthly_rebills"].to_numpy(),
            df["reset_credits_earned"].to_numpy()
        )
    )

    credits_accounted = (
        df["reset_credits_used"]
        + df["reset_credits_expired"]
        + df["reset_credits_remaining"]
    )

    check(
        f"{label}: Reset Credit conservation",
        np.array_equal(
            credits_accounted.to_numpy(),
            df["reset_credits_earned"].to_numpy()
        )
    )

result_level_audit(
    one_full,
    1,
    "One-account full sample"
)

result_level_audit(
    five_full,
    5,
    "Five-account full sample"
)

result_level_audit(
    five_first,
    5,
    "Five-account first 5 years"
)

result_level_audit(
    five_second,
    5,
    "Five-account second 5 years"
)

# ============================================================
# E. FULL-SAMPLE DETAILED BILLING AUDIT
# ============================================================

print()
print("=" * 70)
print("E. FULL-SAMPLE BILLING LOG AUDIT")
print("=" * 70)

event_counts = (
    billing.groupby(
        ["simulation", "event"]
    )
    .size()
    .unstack(fill_value=0)
)

for col in [
    "COMBINE_PURCHASE",
    "MONTHLY_REBILL",
    "PAID_RESET",
    "BACK2FUNDED_REACTIVATION",
    "RESET_CREDIT_USED",
    "RESET_CREDIT_EXPIRED",
]:
    if col not in event_counts.columns:
        event_counts[col] = 0

q = one_full[[
    "simulation",
    "combine_subscriptions_purchased",
    "monthly_rebills",
    "paid_resets",
    "back2funded_reactivations",
    "reset_credits_used",
    "reset_credits_expired",
]].merge(
    event_counts.reset_index(),
    on="simulation",
    how="left"
).fillna(0)

check(
    "Billing events reconcile to result counts",
    (
        (
            q["combine_subscriptions_purchased"]
            == q["COMBINE_PURCHASE"]
        )
        & (
            q["monthly_rebills"]
            == q["MONTHLY_REBILL"]
        )
        & (
            q["paid_resets"]
            == q["PAID_RESET"]
        )
        & (
            q["back2funded_reactivations"]
            == q["BACK2FUNDED_REACTIVATION"]
        )
        & (
            q["reset_credits_used"]
            == q["RESET_CREDIT_USED"]
        )
        & (
            q["reset_credits_expired"]
            == q["RESET_CREDIT_EXPIRED"]
        )
    ).all()
)

rebills = billing[
    billing["event"] == "MONTHLY_REBILL"
].copy()

purchases = billing[
    billing["event"] == "COMBINE_PURCHASE"
].copy()

check(
    "No duplicate same-date rebills on same Combine",
    not rebills.duplicated(
        subset=[
            "simulation",
            "combine_id",
            "date"
        ]
    ).any()
)

check(
    "Every monthly rebill charge = $85",
    (
        rebills["amount"]
        .sub(COMBINE_COST)
        .abs()
        <= TOL
    ).all()
)

check(
    "Every new Combine purchase charge = $85",
    (
        purchases["amount"]
        .sub(COMBINE_COST)
        .abs()
        <= TOL
    ).all()
)

# ============================================================
# F. FULL-SAMPLE OPTION 2 PAYOUT AUDIT
# ============================================================

print()
print("=" * 70)
print("F. FULL-SAMPLE OPTION 2 PAYOUT AUDIT")
print("=" * 70)

payouts = events[
    events["event"] == "PAYOUT"
].copy()

required_payout_cols = [
    "trading_days_before_payout",
    "best_day_before_payout",
    "cycle_profit_before_payout",
    "consistency_before_payout",
    "balance_before_payout",
    "gross_payout",
    "trader_payout",
]

missing_cols = [
    c for c in required_payout_cols
    if c not in payouts.columns
]

check(
    "Payout log contains all Option 2 audit fields",
    len(missing_cols) == 0,
    (
        "missing: " + ", ".join(missing_cols)
        if missing_cols else ""
    )
)

if not missing_cols:
    recomputed_consistency = (
        payouts[
            "best_day_before_payout"
        ].clip(lower=0)
        / payouts[
            "cycle_profit_before_payout"
        ]
    )

    check(
        "Every payout has >=3 trading days",
        (
            payouts[
                "trading_days_before_payout"
            ]
            >= MIN_XFA_TRADING_DAYS
        ).all()
    )

    check(
        "Every payout cycle has positive net profit",
        (
            payouts[
                "cycle_profit_before_payout"
            ] > 0
        ).all()
    )

    check(
        "Stored consistency ratios recalculate exactly",
        close(
            payouts["consistency_before_payout"],
            recomputed_consistency,
            tol=1e-8
        )
    )

    check(
        "Every payout satisfies <=40% consistency",
        (
            payouts["consistency_before_payout"]
            <= MAX_BEST_DAY_SHARE + 1e-10
        ).all()
    )

    check(
        "No gross payout exceeds $6,000 cap",
        (
            payouts["gross_payout"]
            <= PAYOUT_CAP + TOL
        ).all()
    )

    check(
        "No gross payout exceeds 50% of pre-payout balance",
        (
            payouts["gross_payout"]
            <= (
                payouts["balance_before_payout"]
                * 0.50
                + TOL
            )
        ).all()
    )

    check(
        "No payout is below $125",
        (
            payouts["gross_payout"]
            >= MIN_PAYOUT - TOL
        ).all()
    )

    check(
        "Every trader payout is exactly 90% of gross payout",
        close(
            payouts["trader_payout"],
            payouts["gross_payout"]
            * TRADER_SPLIT
        )
    )

# ============================================================
# G. FULL-SAMPLE MLL AUDIT
# ============================================================

print()
print("=" * 70)
print("G. FULL-SAMPLE MLL AUDIT")
print("=" * 70)

trade_mll = mll[
    mll["event"].isin([
        "TRADING_DAY",
        "COMBINE_DLL_LOCKOUT",
        "XFA_DLL_LOCKOUT",
    ])
].copy()

check(
    "MLL never moves downward",
    (
        trade_mll["mll_after"]
        >= trade_mll["mll_before"] - TOL
    ).all()
)

combine_mll = trade_mll[
    trade_mll["phase"] == "COMBINE"
]

xfa_mll = trade_mll[
    trade_mll["phase"] == "XFA"
]

check(
    "Combine MLL never exceeds $50,000 lock",
    (
        combine_mll["mll_after"]
        <= 50_000.0 + TOL
    ).all()
)

check(
    "XFA MLL never exceeds $0 lock",
    (
        xfa_mll["mll_after"]
        <= 0.0 + TOL
    ).all()
)

# ============================================================
# H. FIVE-YEAR SAVED SUMMARY RECONCILIATION
# ============================================================

print()
print("=" * 70)
print("H. SAVED SUMMARY TABLE RECONCILIATION")
print("=" * 70)

def summary_calc(df):
    pnl = df["actual_net_pnl"]
    return {
        "Mean Combine subscriptions purchased":
            df["combine_subscriptions_purchased"].mean(),
        "Mean monthly rebills":
            df["monthly_rebills"].mean(),
        "Mean Reset Credits earned":
            df["reset_credits_earned"].mean(),
        "Mean Reset Credits used":
            df["reset_credits_used"].mean(),
        "Mean Combine passes":
            df["combine_passes"].mean(),
        "Mean XFA accounts earned":
            df["xfa_accounts_earned"].mean(),
        "Mean XFA failures":
            df["xfa_failures"].mean(),
        "Mean payouts":
            df["payouts"].mean(),
        "Mean trader payouts":
            df["trader_payouts"].mean(),
        "Mean account costs":
            df["account_costs"].mean(),
        "TOTAL trader payouts":
            df["trader_payouts"].sum(),
        "TOTAL account costs":
            df["account_costs"].sum(),
        "TOTAL ACTUAL net P&L":
            pnl.sum(),
        "Mean ACTUAL net P&L":
            pnl.mean(),
        "Median ACTUAL net P&L":
            pnl.median(),
        "Std. dev. ACTUAL net P&L":
            pnl.std(),
        "5th percentile ACTUAL net P&L":
            pnl.quantile(.05),
        "95th percentile ACTUAL net P&L":
            pnl.quantile(.95),
        "Profitable simulations":
            int((pnl > 0).sum()),
        "Profitable %":
            (pnl > 0).mean() * 100,
        "Losing simulations":
            int((pnl < 0).sum()),
        "Losing %":
            (pnl < 0).mean() * 100,
    }

summary_sources = {
    "Full sample": five_full,
    "2016-08-08 to 2021-08-06": five_first,
    "2021-08-09 to 2026-08-05": five_second,
}

for period, df in summary_sources.items():
    saved = five_summary[
        five_summary["Period"] == period
    ]

    check(
        f"{period}: summary row exists",
        len(saved) == 1
    )

    if len(saved) != 1:
        continue

    saved = saved.iloc[0]
    calc = summary_calc(df)

    for col, value in calc.items():
        check(
            f"{period}: summary {col}",
            close(saved[col], value),
            f"saved={saved[col]} recalc={value}"
        )

# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 70)

if errors:
    print("FINAL AUDIT STATUS: FAIL")
    print()
    print("FAILED CHECKS:")
    for e in errors:
        print("-", e)
    print()
    print(
        "Do not use the affected results in the paper until "
        "the failed checks are investigated."
    )
else:
    print("FINAL AUDIT STATUS: PASS")
    print()
    print(
        "Raw strategy, exact five-year split, full Topstep "
        "Option 2 accounting, five-account scaling, saved "
        "five-year Topstep results, billing, payout rules, "
        "Reset Credits, MLL behavior, and P&L reconciliation "
        "all passed."
    )

print("=" * 70)
