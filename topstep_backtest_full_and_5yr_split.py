from pathlib import Path
import pandas as pd
import numpy as np
import os

# ============================================================
# TOPSTEP 50K — 10,000 RANDOM NULL STRATEGIES
# CURRENT-RULE MODEL: NO ACTIVATION FEE + PURCHASE-SET DLL
# XFA OPTION 2 / CONSISTENCY
#
# Configuration modeled:
#   - 50K No Activation Fee Trading Combine
#   - Purchase-set $1,000 DLL
#   - Responsible Trading discount:
#         $85 initial / monthly Combine subscription
#   - XFA Standard (Option 1)
#   - $4,000 XFA payout cap unlocked by DLL-at-purchase
#   - 90/10 payout split
#   - Reset Bank credits from monthly rebills
#   - Back2Funded used whenever eligible
#   - TopstepX MNQ commissions/fees included in simulated P&L
#
# IMPORTANT:
#   This model applies the CURRENT 2026 Topstep rule framework
#   to the historical randomized paths. It is NOT a historical
#   reconstruction of Topstep's rules in 2016-2025.
#
#   Discretionary Live Funded Account call-ups, discretionary
#   compliance/prohibited-conduct reviews, taxes, and slippage
#   cannot be deterministically modeled from public rules and
#   are therefore outside this simulation.
# ============================================================


# ============================================================
# FILES / STRATEGY BUILD
# ============================================================

NQ_FILE = "NQ_data.csv"

# Strategy-specific outputs
PATH_FILE = "mnq_10am_100tick_8con_10000_paths.npy"
DAY_FILE = "mnq_10am_100tick_8con_clean_days.csv"

# Topstep Option 2 outputs
OUTPUT_FILE = "topstep_option2_10am_100tick_8con_10000_results.csv"
ACCOUNT_LOG_FILE = "topstep_option2_10am_100tick_8con_account_log.csv"
MLL_AUDIT_FILE = "topstep_option2_10am_100tick_8con_mll_audit.csv"
BILLING_AUDIT_FILE = "topstep_option2_10am_100tick_8con_billing_audit.csv"

NUM_SIMULATIONS = 10_000
BASE_SEED = 42

REQUIRED_START_TIME = "09:30:00"
ENTRY_TIME_REQUIRED = "10:00:00"
REQUIRED_END_TIME = "16:00:00"


# ============================================================
# UNDERLYING STRATEGY
# ============================================================

CONTRACTS = 8
TICK_SIZE = 0.25
TICKS = 100
TP_SL_POINTS = TICK_SIZE * TICKS

MNQ_DOLLARS_PER_POINT = 2.0
GROSS_TP_SL_DOLLARS = (
    TP_SL_POINTS
    * MNQ_DOLLARS_PER_POINT
    * CONTRACTS
)

# TopstepX current MNQ round-turn cost:
# $1.22 per contract round-turn.
MNQ_RT_COST_PER_CONTRACT = 1.22
ROUND_TURN_COST = (
    MNQ_RT_COST_PER_CONTRACT
    * CONTRACTS
)
ENTRY_SIDE_COST = ROUND_TURN_COST / 2.0

# Position-size checks.
COMBINE_MAX_MICROS_50K = 50
XFA_STARTING_MAX_MICROS_50K = 20

if CONTRACTS > COMBINE_MAX_MICROS_50K:
    raise ValueError(
        "Strategy exceeds 50K Trading Combine max micros."
    )

if CONTRACTS > XFA_STARTING_MAX_MICROS_50K:
    raise ValueError(
        "Strategy exceeds starting 50K XFA Scaling Plan."
    )


# ============================================================
# 50K TRADING COMBINE
# ============================================================

COMBINE_STARTING_BALANCE = 50_000.00
COMBINE_PROFIT_TARGET = 3_000.00
COMBINE_MLL_DISTANCE = 2_000.00

# Purchase-set DLL used to obtain the responsible-trading
# monthly discount and doubled XFA payout cap.
COMBINE_DLL = 1_000.00

COMBINE_MIN_TRADING_DAYS = 2


# ============================================================
# XFA CONSISTENCY / OPTION 2
# ============================================================

XFA_STARTING_BALANCE = 0.00
XFA_MLL_DISTANCE = 2_000.00
XFA_DLL = 1_000.00

# Option 2 / Consistency:
# - At least 3 trading days with >= 1 trade per day
# - Largest winning day <= 40% of total net profit
# - Consistency and 3-day count reset after every payout
XFA_CONSISTENCY_MIN_TRADING_DAYS = 3
XFA_CONSISTENCY_MAX_BEST_DAY_SHARE = 0.40

PAYOUT_CAP = 6_000.00
MIN_PAYOUT_REQUEST = 125.00
TRADER_SPLIT = 0.90


# ============================================================
# BILLING / RESET BANK
# ============================================================

COMBINE_MONTHLY_COST = 85.00
PAID_RESET_COST = 95.00

RESET_CREDITS_PER_REBILL = 1
RESET_CREDIT_LIFE_YEARS = 1

BACK2FUNDED_COST = 549.00
BACK2FUNDED_MAX_REACTIVATIONS = 2
USE_BACK2FUNDED = False

USE_RESET_CREDIT_FIRST = True
PREFER_NEW_COMBINE_OVER_PAID_RESET = True


# ============================================================
# LOAD 1-MINUTE NQ DATA
# ============================================================

if not Path(NQ_FILE).exists():
    print(f"ERROR: {NQ_FILE} not found.")
    raise SystemExit

nq_full = pd.read_csv(
    NQ_FILE,
    sep=";",
    header=None,
    names=[
        "date",
        "time",
        "O",
        "H",
        "L",
        "C",
        "volume"
    ]
)

for col in ["O", "H", "L", "C", "volume"]:
    nq_full[col] = pd.to_numeric(
        nq_full[col],
        errors="coerce"
    )

nq_full["datetime"] = pd.to_datetime(
    nq_full["date"].astype(str)
    + " "
    + nq_full["time"].astype(str),
    errors="coerce"
)

nq_full = nq_full.dropna(
    subset=[
        "datetime",
        "O",
        "H",
        "L",
        "C"
    ]
)

nq_full = (
    nq_full
    .sort_values("datetime")
    .reset_index(drop=True)
)


# ============================================================
# BUILD 10:00 STRATEGY DAY RESULTS
# ============================================================

print("==============================")
print("BUILDING 10:00 STRATEGY DAYS")
print("==============================")

day_results = []
ambiguous_days = []
excluded_nonfull = 0
excluded_no_10am = 0

for date_key, day in nq_full.groupby(
    nq_full["datetime"].dt.date
):

    day = (
        day
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    first_time = (
        day.iloc[0]["datetime"]
        .strftime("%H:%M:%S")
    )
    last_time = (
        day.iloc[-1]["datetime"]
        .strftime("%H:%M:%S")
    )

    if (
        first_time != REQUIRED_START_TIME
        or last_time != REQUIRED_END_TIME
    ):
        excluded_nonfull += 1
        continue

    entry_candidates = day[
        day["datetime"].dt.strftime("%H:%M:%S")
        == ENTRY_TIME_REQUIRED
    ]

    if len(entry_candidates) != 1:
        excluded_no_10am += 1
        continue

    entry_bar = entry_candidates.iloc[0]
    entry = float(entry_bar["O"])
    entry_time = entry_bar["datetime"]

    long_target = entry + TP_SL_POINTS
    long_stop = entry - TP_SL_POINTS

    short_target = entry - TP_SL_POINTS
    short_stop = entry + TP_SL_POINTS

    post_entry = day[
        day["datetime"] > entry_time
    ]

    # -------------------------
    # LONG
    # -------------------------

    long_result = None
    long_exit_price = None
    long_exit_time = None

    for _, bar in post_entry.iterrows():

        hit_target = (
            bar["H"] >= long_target
        )
        hit_stop = (
            bar["L"] <= long_stop
        )

        if hit_target and hit_stop:
            long_result = "AMBIGUOUS"
            break

        if hit_target:
            long_result = "WIN"
            long_exit_price = long_target
            long_exit_time = bar["datetime"]
            break

        if hit_stop:
            long_result = "LOSS"
            long_exit_price = long_stop
            long_exit_time = bar["datetime"]
            break

    if long_result is None:
        final_close = float(
            day.iloc[-1]["C"]
        )
        long_result = "TIME_EXIT"
        long_exit_price = final_close
        long_exit_time = (
            day.iloc[-1]["datetime"]
        )
        long_pnl = (
            (final_close - entry)
            * MNQ_DOLLARS_PER_POINT
            * CONTRACTS
        )

    elif long_result == "WIN":
        long_pnl = (
            GROSS_TP_SL_DOLLARS
        )

    elif long_result == "LOSS":
        long_pnl = (
            -GROSS_TP_SL_DOLLARS
        )

    else:
        long_pnl = np.nan

    # -------------------------
    # SHORT
    # -------------------------

    short_result = None
    short_exit_price = None
    short_exit_time = None

    for _, bar in post_entry.iterrows():

        hit_target = (
            bar["L"] <= short_target
        )
        hit_stop = (
            bar["H"] >= short_stop
        )

        if hit_target and hit_stop:
            short_result = "AMBIGUOUS"
            break

        if hit_target:
            short_result = "WIN"
            short_exit_price = short_target
            short_exit_time = bar["datetime"]
            break

        if hit_stop:
            short_result = "LOSS"
            short_exit_price = short_stop
            short_exit_time = bar["datetime"]
            break

    if short_result is None:
        final_close = float(
            day.iloc[-1]["C"]
        )
        short_result = "TIME_EXIT"
        short_exit_price = final_close
        short_exit_time = (
            day.iloc[-1]["datetime"]
        )
        short_pnl = (
            (entry - final_close)
            * MNQ_DOLLARS_PER_POINT
            * CONTRACTS
        )

    elif short_result == "WIN":
        short_pnl = (
            GROSS_TP_SL_DOLLARS
        )

    elif short_result == "LOSS":
        short_pnl = (
            -GROSS_TP_SL_DOLLARS
        )

    else:
        short_pnl = np.nan

    if (
        long_result == "AMBIGUOUS"
        or short_result == "AMBIGUOUS"
    ):
        ambiguous_days.append(
            str(date_key)
        )

    day_results.append({
        "date": str(date_key),
        "entry_time": entry_time,
        "entry": entry,

        "long_pnl": long_pnl,
        "short_pnl": short_pnl,

        "long_result": long_result,
        "short_result": short_result,

        "long_exit_price":
            long_exit_price,
        "short_exit_price":
            short_exit_price,

        "long_exit_time":
            long_exit_time,
        "short_exit_time":
            short_exit_time
    })


day_results = pd.DataFrame(
    day_results
)

clean_days = day_results[
    day_results["long_pnl"].notna()
    & day_results["short_pnl"].notna()
].copy().reset_index(drop=True)

clean_days.to_csv(
    DAY_FILE,
    index=False
)

print(
    f"Full-session exclusions: "
    f"{excluded_nonfull:,}"
)
print(
    f"Full sessions missing 10:00 bar: "
    f"{excluded_no_10am:,}"
)
print(
    f"Ambiguous days excluded: "
    f"{len(ambiguous_days):,}"
)
print(
    f"Final clean trading days: "
    f"{len(clean_days):,}"
)
print(
    f"Gross TP/SL value: "
    f"${GROSS_TP_SL_DOLLARS:,.2f}"
)
print(
    f"8-MNQ round-turn commission: "
    f"${ROUND_TURN_COST:,.2f}"
)


# ============================================================
# GENERATE 10,000 RANDOM LONG/SHORT PATHS
# ============================================================

rng = np.random.default_rng(
    BASE_SEED
)

paths = rng.integers(
    0,
    2,
    size=(
        NUM_SIMULATIONS,
        len(clean_days)
    ),
    dtype=np.int8
)

np.save(
    PATH_FILE,
    paths
)

print()
print("==============================")
print("VERIFYING RANDOM PATH DATA")
print("==============================")

print(
    f"Random path shape: "
    f"{paths.shape}"
)

print(
    f"Clean trading days: "
    f"{len(clean_days):,}"
)

if paths.shape[0] != NUM_SIMULATIONS:
    raise ValueError(
        f"Expected {NUM_SIMULATIONS:,} simulations "
        f"but found {paths.shape[0]:,}."
    )

if paths.shape[1] != len(clean_days):
    raise ValueError(
        "Random-path columns do not match clean trading days."
    )


# ============================================================
# PREPARE CLEAN-DAY DATA
# ============================================================

clean_days["date_dt"] = pd.to_datetime(
    clean_days["date"]
).dt.normalize()

dates = clean_days["date_dt"].to_numpy()

long_gross_pnl = clean_days[
    "long_pnl"
].to_numpy(dtype=float)

short_gross_pnl = clean_days[
    "short_pnl"
].to_numpy(dtype=float)

# Topstep simulated Net P&L includes commissions/fees.
long_net_pnl = long_gross_pnl - ROUND_TURN_COST
short_net_pnl = short_gross_pnl - ROUND_TURN_COST


# ============================================================
# PRECOMPUTE INTRADAY ADVERSE EXCURSION
# ============================================================
#
# Topstep monitors MLL/DLL in real time using unrealized P&L.
# The old day-level simulator only checked EOD balances.
#
# We therefore use the original 1-minute NQ data to compute,
# for every clean day:
#
#   long_mae  = worst unrealized LONG P&L before exit
#   short_mae = worst unrealized SHORT P&L before exit
#
# The strategy uses a 50-point stop, so adverse excursion is
# capped at the modeled stop fill (-$400 gross). This preserves
# the same fill convention as the raw backtest.
#
# For a target hit inside a one-minute exit bar, the precise
# intrabar ordering of that bar's favorable/adverse extremes
# is unknowable. This audit uses the full exit-bar range, which
# is conservative for MLL/DLL purposes: it may count an adverse
# excursion that happened after the target was reached, but it
# will not miss one that could have occurred before the exit.
# ============================================================

print()
print("==============================")
print("PRECOMPUTING INTRADAY RISK")
print("==============================")

nq = pd.read_csv(
    NQ_FILE,
    sep=";",
    header=None,
    names=[
        "date",
        "time",
        "O",
        "H",
        "L",
        "C",
        "volume"
    ]
)

for col in ["O", "H", "L", "C"]:
    nq[col] = pd.to_numeric(
        nq[col],
        errors="coerce"
    )

nq["datetime"] = pd.to_datetime(
    nq["date"].astype(str)
    + " "
    + nq["time"].astype(str),
    errors="coerce"
)

nq = nq.dropna(
    subset=[
        "datetime",
        "O",
        "H",
        "L",
        "C"
    ]
)

nq["date_dt"] = nq[
    "datetime"
].dt.normalize()

clean_date_set = set(
    clean_days["date_dt"]
)

nq = nq[
    nq["date_dt"].isin(clean_date_set)
].copy()

day_groups = {
    d: g.sort_values("datetime").reset_index(drop=True)
    for d, g in nq.groupby("date_dt")
}

long_mae = np.empty(
    len(clean_days),
    dtype=float
)

short_mae = np.empty(
    len(clean_days),
    dtype=float
)

for i, row in clean_days.iterrows():

    d = row["date_dt"]

    if d not in day_groups:
        raise ValueError(
            f"No 1-minute source data found for {d.date()}."
        )

    day = day_groups[d]

    entry = float(row["entry"])
    entry_time = pd.to_datetime(row["entry_time"])

    long_exit_time = pd.to_datetime(
        row["long_exit_time"]
    )

    short_exit_time = pd.to_datetime(
        row["short_exit_time"]
    )

    # Match the validated raw backtest convention:
    # post-entry bars are strictly after the 09:30 entry bar.
    long_bars = day[
        (day["datetime"] > entry_time)
        & (day["datetime"] <= long_exit_time)
    ]

    short_bars = day[
        (day["datetime"] > entry_time)
        & (day["datetime"] <= short_exit_time)
    ]

    if len(long_bars) == 0:
        long_raw_mae = 0.0
    else:
        worst_long_price = float(
            long_bars["L"].min()
        )

        long_raw_mae = (
            (worst_long_price - entry)
            * MNQ_DOLLARS_PER_POINT
            * CONTRACTS
        )

    if len(short_bars) == 0:
        short_raw_mae = 0.0
    else:
        worst_short_price = float(
            short_bars["H"].max()
        )

        short_raw_mae = (
            (entry - worst_short_price)
            * MNQ_DOLLARS_PER_POINT
            * CONTRACTS
        )

    # Stop order means modeled gross adverse P&L cannot
    # continue below -$400.
    long_mae[i] = max(
        long_raw_mae,
        -GROSS_TP_SL_DOLLARS
    )

    short_mae[i] = max(
        short_raw_mae,
        -GROSS_TP_SL_DOLLARS
    )


print(
    f"Intraday risk records: "
    f"{len(long_mae):,} LONG + "
    f"{len(short_mae):,} SHORT"
)


# ============================================================
# COMBINE CONSISTENCY TARGET
# ============================================================
#
# Best day should remain below 50% of the profit target.
# If it is too large, the amount required to satisfy the
# consistency relationship rises accordingly.
# ============================================================

def combine_target(best_day):

    if best_day <= 0:
        return COMBINE_PROFIT_TARGET

    return max(
        COMBINE_PROFIT_TARGET,
        best_day / 0.50
    )


# ============================================================
# RUN ONE SIMULATION
# ============================================================

def run_simulation(sim_index):

    directions = paths[sim_index]

    daily_pnl = np.where(
        directions == 1,
        long_net_pnl,
        short_net_pnl
    )

    daily_mae = np.where(
        directions == 1,
        long_mae,
        short_mae
    )

    # --------------------------------------------------------
    # OVERALL STATE
    # --------------------------------------------------------

    phase = "COMBINE"

    account_costs = 0.0
    payout_total = 0.0

    accounts_blown = 0
    combine_passes = 0
    xfa_failures = 0
    payouts = 0

    combine_subscriptions_purchased = 0
    monthly_rebills = 0
    paid_resets = 0
    reset_credits_earned = 0
    reset_credits_used = 0
    reset_credits_expired = 0

    xfa_accounts_earned = 0
    back2funded_reactivations = 0

    account_events = []
    mll_audit_events = []
    billing_events = []

    # Reset Bank persists at profile level.
    reset_credit_dates = []

    # --------------------------------------------------------
    # COMBINE STATE
    # --------------------------------------------------------

    combine_id = 0
    balance = COMBINE_STARTING_BALANCE
    combine_mll = (
        COMBINE_STARTING_BALANCE
        - COMBINE_MLL_DISTANCE
    )
    combine_best_day = 0.0
    combine_trading_days = 0
    next_rebill_date = None

    # --------------------------------------------------------
    # XFA STATE
    # --------------------------------------------------------

    xfa_id = 0
    xfa_balance = XFA_STARTING_BALANCE
    xfa_mll = -XFA_MLL_DISTANCE

    xfa_cycle_trading_days = 0
    xfa_cycle_profit = 0.0
    xfa_cycle_best_day = 0.0

    # Per-XFA payout state.
    xfa_payouts_this_account = 0

    # Back2Funded count tied to the current earned XFA.
    xfa_reactivations_used = 0

    # --------------------------------------------------------
    # RESET CREDIT HELPERS
    # --------------------------------------------------------

    def expire_reset_credits(current_date):

        nonlocal reset_credits_expired
        nonlocal reset_credit_dates

        current_date = pd.Timestamp(
            current_date
        ).normalize()

        kept = []

        for issue_date in reset_credit_dates:

            expiry_date = (
                pd.Timestamp(issue_date)
                + pd.DateOffset(
                    years=RESET_CREDIT_LIFE_YEARS
                )
            )

            if expiry_date <= current_date:

                reset_credits_expired += 1

                billing_events.append({
                    "simulation": sim_index + 1,
                    "date": current_date,
                    "event": "RESET_CREDIT_EXPIRED",
                    "amount": 0.0,
                    "combine_id": combine_id,
                    "credit_issue_date":
                        pd.Timestamp(issue_date)
                })

            else:
                kept.append(
                    pd.Timestamp(issue_date)
                )

        reset_credit_dates = kept

    def use_oldest_reset_credit(current_date):

        nonlocal reset_credits_used
        nonlocal reset_credit_dates

        expire_reset_credits(
            current_date
        )

        if len(reset_credit_dates) == 0:
            return False

        reset_credit_dates.sort()

        credit_date = (
            reset_credit_dates.pop(0)
        )

        reset_credits_used += 1

        billing_events.append({
            "simulation": sim_index + 1,
            "date": pd.Timestamp(current_date),
            "event": "RESET_CREDIT_USED",
            "amount": 0.0,
            "combine_id": combine_id,
            "credit_issue_date":
                pd.Timestamp(credit_date)
        })

        return True

    # --------------------------------------------------------
    # START NEW COMBINE SUBSCRIPTION
    # --------------------------------------------------------

    def purchase_new_combine(
        reason,
        current_date
    ):

        nonlocal phase
        nonlocal account_costs

        nonlocal combine_id
        nonlocal combine_subscriptions_purchased

        nonlocal balance
        nonlocal combine_mll
        nonlocal combine_best_day
        nonlocal combine_trading_days
        nonlocal next_rebill_date

        combine_id += 1
        combine_subscriptions_purchased += 1

        account_costs += (
            COMBINE_MONTHLY_COST
        )

        phase = "COMBINE"

        balance = (
            COMBINE_STARTING_BALANCE
        )

        combine_mll = (
            COMBINE_STARTING_BALANCE
            - COMBINE_MLL_DISTANCE
        )

        combine_best_day = 0.0
        combine_trading_days = 0

        current_date = pd.Timestamp(
            current_date
        ).normalize()

        next_rebill_date = (
            current_date
            + pd.Timedelta(days=30)
        )

        account_events.append({
            "simulation": sim_index + 1,
            "combine_id": combine_id,
            "xfa_id": np.nan,
            "event": "NEW_COMBINE_PURCHASE",
            "date": current_date,
            "reason": reason,
            "balance": balance,
            "mll": combine_mll
        })

        billing_events.append({
            "simulation": sim_index + 1,
            "date": current_date,
            "event": "COMBINE_PURCHASE",
            "amount": COMBINE_MONTHLY_COST,
            "combine_id": combine_id,
            "credit_issue_date": pd.NaT
        })

        mll_audit_events.append({
            "simulation": sim_index + 1,
            "combine_id": combine_id,
            "xfa_id": np.nan,
            "date": current_date,
            "phase": "COMBINE",
            "event": "NEW_COMBINE_PURCHASE",
            "balance_before": np.nan,
            "balance_after": balance,
            "mll_before": np.nan,
            "mll_after": combine_mll
        })

    # --------------------------------------------------------
    # PROCESS MONTHLY REBILLS
    # --------------------------------------------------------

    def process_combine_rebills(
        current_date
    ):

        nonlocal account_costs
        nonlocal monthly_rebills
        nonlocal reset_credits_earned
        nonlocal next_rebill_date

        current_date = pd.Timestamp(
            current_date
        ).normalize()

        expire_reset_credits(
            current_date
        )

        while (
            phase == "COMBINE"
            and next_rebill_date is not None
            and next_rebill_date <= current_date
        ):

            rebill_date = pd.Timestamp(
                next_rebill_date
            ).normalize()

            account_costs += (
                COMBINE_MONTHLY_COST
            )

            monthly_rebills += 1

            for _ in range(
                RESET_CREDITS_PER_REBILL
            ):
                reset_credit_dates.append(
                    rebill_date
                )

                reset_credits_earned += 1

            billing_events.append({
                "simulation": sim_index + 1,
                "date": rebill_date,
                "event": "MONTHLY_REBILL",
                "amount": COMBINE_MONTHLY_COST,
                "combine_id": combine_id,
                "credit_issue_date": rebill_date
            })

            next_rebill_date = (
                rebill_date
                + pd.Timedelta(days=30)
            )

    # --------------------------------------------------------
    # RESET FAILED COMBINE
    # --------------------------------------------------------

    def reset_current_combine(
        current_date,
        reason
    ):

        nonlocal account_costs
        nonlocal paid_resets

        nonlocal balance
        nonlocal combine_mll
        nonlocal combine_best_day
        nonlocal combine_trading_days
        nonlocal next_rebill_date

        current_date = pd.Timestamp(
            current_date
        ).normalize()

        credit_used = False

        if USE_RESET_CREDIT_FIRST:
            credit_used = (
                use_oldest_reset_credit(
                    current_date
                )
            )

        if not credit_used:

            if PREFER_NEW_COMBINE_OVER_PAID_RESET:

                # A new matching Combine is cheaper than a
                # purchased Reset under this configuration.
                # Cancel/replace the failed subscription and
                # start a fresh 30-day Combine billing window.
                purchase_new_combine(
                    "Replace failed Combine: " + reason,
                    current_date
                )

                return

            account_costs += (
                PAID_RESET_COST
            )

            paid_resets += 1

            billing_events.append({
                "simulation": sim_index + 1,
                "date": current_date,
                "event": "PAID_RESET",
                "amount": PAID_RESET_COST,
                "combine_id": combine_id,
                "credit_issue_date": pd.NaT
            })

        # Purchased Reset OR applied credit:
        # same active subscription, fresh account state,
        # and Rebill moves 30 days from reset date.
        balance = (
            COMBINE_STARTING_BALANCE
        )

        combine_mll = (
            COMBINE_STARTING_BALANCE
            - COMBINE_MLL_DISTANCE
        )

        combine_best_day = 0.0
        combine_trading_days = 0

        next_rebill_date = (
            current_date
            + pd.Timedelta(days=30)
        )

        account_events.append({
            "simulation": sim_index + 1,
            "combine_id": combine_id,
            "xfa_id": np.nan,
            "event": (
                "COMBINE_RESET_CREDIT"
                if credit_used
                else "COMBINE_RESET_PAID"
            ),
            "date": current_date,
            "reason": reason,
            "balance": balance,
            "mll": combine_mll
        })

        mll_audit_events.append({
            "simulation": sim_index + 1,
            "combine_id": combine_id,
            "xfa_id": np.nan,
            "date": current_date,
            "phase": "COMBINE",
            "event": (
                "COMBINE_RESET_CREDIT"
                if credit_used
                else "COMBINE_RESET_PAID"
            ),
            "balance_before": np.nan,
            "balance_after": balance,
            "mll_before": np.nan,
            "mll_after": combine_mll
        })

    # --------------------------------------------------------
    # START BRAND-NEW XFA AFTER COMBINE PASS
    # --------------------------------------------------------

    def start_new_xfa(
        current_date
    ):

        nonlocal phase
        nonlocal xfa_id
        nonlocal xfa_accounts_earned

        nonlocal xfa_balance
        nonlocal xfa_mll
        nonlocal xfa_cycle_trading_days
        nonlocal xfa_cycle_profit
        nonlocal xfa_cycle_best_day
        nonlocal xfa_payouts_this_account
        nonlocal xfa_reactivations_used
        nonlocal next_rebill_date

        xfa_id += 1
        xfa_accounts_earned += 1

        phase = "XFA"

        # Combine subscription cancels upon pass.
        next_rebill_date = None

        xfa_balance = (
            XFA_STARTING_BALANCE
        )

        xfa_mll = (
            -XFA_MLL_DISTANCE
        )

        xfa_cycle_trading_days = 0
        xfa_cycle_profit = 0.0
        xfa_cycle_best_day = 0.0
        xfa_payouts_this_account = 0
        xfa_reactivations_used = 0

        account_events.append({
            "simulation": sim_index + 1,
            "combine_id": combine_id,
            "xfa_id": xfa_id,
            "event": "XFA_START",
            "date": pd.Timestamp(current_date),
            "reason": "Combine passed",
            "balance": xfa_balance,
            "mll": xfa_mll
        })

        mll_audit_events.append({
            "simulation": sim_index + 1,
            "combine_id": combine_id,
            "xfa_id": xfa_id,
            "date": pd.Timestamp(current_date),
            "phase": "XFA",
            "event": "XFA_START",
            "balance_before": np.nan,
            "balance_after": xfa_balance,
            "mll_before": np.nan,
            "mll_after": xfa_mll
        })

    # --------------------------------------------------------
    # BACK2FUNDED REACTIVATION
    # --------------------------------------------------------

    def reactivate_xfa(
        current_date
    ):

        nonlocal account_costs
        nonlocal back2funded_reactivations
        nonlocal xfa_reactivations_used

        nonlocal xfa_balance
        nonlocal xfa_mll
        nonlocal xfa_cycle_trading_days
        nonlocal xfa_cycle_profit
        nonlocal xfa_cycle_best_day
        nonlocal xfa_payouts_this_account

        account_costs += (
            BACK2FUNDED_COST
        )

        back2funded_reactivations += 1
        xfa_reactivations_used += 1

        xfa_balance = (
            XFA_STARTING_BALANCE
        )

        xfa_mll = (
            -XFA_MLL_DISTANCE
        )

        xfa_cycle_trading_days = 0
        xfa_cycle_profit = 0.0
        xfa_cycle_best_day = 0.0
        xfa_payouts_this_account = 0

        billing_events.append({
            "simulation": sim_index + 1,
            "date": pd.Timestamp(current_date),
            "event": "BACK2FUNDED_REACTIVATION",
            "amount": BACK2FUNDED_COST,
            "combine_id": combine_id,
            "credit_issue_date": pd.NaT
        })

        account_events.append({
            "simulation": sim_index + 1,
            "combine_id": combine_id,
            "xfa_id": xfa_id,
            "event": "XFA_REACTIVATED",
            "date": pd.Timestamp(current_date),
            "reason": "Back2Funded",
            "balance": xfa_balance,
            "mll": xfa_mll
        })

        mll_audit_events.append({
            "simulation": sim_index + 1,
            "combine_id": combine_id,
            "xfa_id": xfa_id,
            "date": pd.Timestamp(current_date),
            "phase": "XFA",
            "event": "XFA_REACTIVATED",
            "balance_before": np.nan,
            "balance_after": xfa_balance,
            "mll_before": np.nan,
            "mll_after": xfa_mll
        })

    # --------------------------------------------------------
    # INITIAL COMBINE
    # --------------------------------------------------------

    purchase_new_combine(
        "Initial Trading Combine",
        dates[0]
    )

    # ========================================================
    # PROCESS TRADING DAYS
    # ========================================================

    for day_index, pnl in enumerate(
        daily_pnl
    ):

        pnl = float(pnl)

        # Gross adverse excursion during the trade.
        gross_mae = float(
            daily_mae[day_index]
        )

        # Real-time net adverse P&L includes entry-side fees.
        net_intraday_mae = (
            gross_mae
            - ENTRY_SIDE_COST
        )

        date = pd.Timestamp(
            dates[day_index]
        ).normalize()

        # ====================================================
        # COMBINE
        # ====================================================

        if phase == "COMBINE":

            process_combine_rebills(
                date
            )

            old_balance = balance
            old_mll = combine_mll

            # ------------------------------------------------
            # REAL-TIME RISK CHECK
            # ------------------------------------------------

            # MLL allowable daily adverse move from current
            # starting balance.
            mll_loss_threshold = (
                combine_mll
                - old_balance
            )

            dll_loss_threshold = (
                -COMBINE_DLL
            )

            # Worst net P&L observed for the day must include
            # both unrealized adverse excursion (entry-side fees)
            # and the final realized net P&L (full round-turn fees).
            # This prevents a small commission-induced realized-loss
            # breach from being missed after the position closes.
            day_worst_net_pnl = min(
                net_intraday_mae,
                pnl
            )

            hit_mll = (
                day_worst_net_pnl
                <= mll_loss_threshold
            )

            hit_dll = (
                day_worst_net_pnl
                <= dll_loss_threshold
            )

            # If both could be touched, whichever threshold is
            # closer to zero triggers first on a monotonic
            # adverse move.
            if hit_mll and (
                not hit_dll
                or mll_loss_threshold
                >= dll_loss_threshold
            ):

                accounts_blown += 1

                account_events.append({
                    "simulation": sim_index + 1,
                    "combine_id": combine_id,
                    "xfa_id": np.nan,
                    "event": "COMBINE_FAIL_MLL",
                    "date": date,
                    "reason": "Maximum Loss Limit",
                    "balance": combine_mll,
                    "mll": combine_mll
                })

                mll_audit_events.append({
                    "simulation": sim_index + 1,
                    "combine_id": combine_id,
                    "xfa_id": np.nan,
                    "date": date,
                    "phase": "COMBINE",
                    "event": "COMBINE_FAIL_MLL",
                    "balance_before": old_balance,
                    "balance_after": combine_mll,
                    "mll_before": old_mll,
                    "mll_after": old_mll
                })

                reset_current_combine(
                    date,
                    "Combine MLL breach"
                )

                continue

            if hit_dll:

                # DLL is NOT a failed account.
                # Positions are flattened and trader is blocked
                # for the rest of that session only.
                #
                # Under this strategy's $400 stop, this branch
                # should not bind. We approximate the forced
                # liquidation at the DLL threshold if it does.
                dll_day_pnl = (
                    -COMBINE_DLL
                    - ENTRY_SIDE_COST
                )

                balance = (
                    old_balance
                    + dll_day_pnl
                )

                combine_trading_days += 1

                account_events.append({
                    "simulation": sim_index + 1,
                    "combine_id": combine_id,
                    "xfa_id": np.nan,
                    "event": "COMBINE_DLL_LOCKOUT",
                    "date": date,
                    "reason": "Daily Loss Limit",
                    "balance": balance,
                    "mll": combine_mll
                })

                # EOD MLL trails, capped at starting balance.
                candidate_mll = (
                    balance
                    - COMBINE_MLL_DISTANCE
                )

                combine_mll = max(
                    combine_mll,
                    min(
                        COMBINE_STARTING_BALANCE,
                        candidate_mll
                    )
                )

                mll_audit_events.append({
                    "simulation": sim_index + 1,
                    "combine_id": combine_id,
                    "xfa_id": np.nan,
                    "date": date,
                    "phase": "COMBINE",
                    "event": "COMBINE_DLL_LOCKOUT",
                    "balance_before": old_balance,
                    "balance_after": balance,
                    "mll_before": old_mll,
                    "mll_after": combine_mll
                })

                continue

            # ------------------------------------------------
            # ACCEPT NORMAL DAY
            # ------------------------------------------------

            balance = (
                old_balance
                + pnl
            )

            combine_trading_days += 1

            if pnl > combine_best_day:
                combine_best_day = pnl

            # EOD trailing MLL; cannot rise above $50,000.
            candidate_mll = (
                balance
                - COMBINE_MLL_DISTANCE
            )

            combine_mll = max(
                combine_mll,
                min(
                    COMBINE_STARTING_BALANCE,
                    candidate_mll
                )
            )

            mll_audit_events.append({
                "simulation": sim_index + 1,
                "combine_id": combine_id,
                "xfa_id": np.nan,
                "date": date,
                "phase": "COMBINE",
                "event": "TRADING_DAY",
                "balance_before": old_balance,
                "balance_after": balance,
                "mll_before": old_mll,
                "mll_after": combine_mll
            })

            # ------------------------------------------------
            # PROFIT + CONSISTENCY PASS
            # ------------------------------------------------

            total_profit = (
                balance
                - COMBINE_STARTING_BALANCE
            )

            required_profit = (
                combine_target(
                    combine_best_day
                )
            )

            if (
                total_profit
                >= required_profit
                and combine_best_day
                <= required_profit * 0.50
                and combine_trading_days
                >= COMBINE_MIN_TRADING_DAYS
            ):

                combine_passes += 1

                account_events.append({
                    "simulation": sim_index + 1,
                    "combine_id": combine_id,
                    "xfa_id": np.nan,
                    "event": "COMBINE_PASS",
                    "date": date,
                    "reason":
                        "Profit target + consistency",
                    "combine_balance": balance,
                    "combine_mll": combine_mll,
                    "best_day": combine_best_day,
                    "required_profit": required_profit,
                    "trading_days":
                        combine_trading_days
                })

                start_new_xfa(
                    date
                )

        # ====================================================
        # XFA CONSISTENCY / OPTION 2
        # ====================================================

        elif phase == "XFA":

            old_balance = xfa_balance
            old_mll = xfa_mll

            # ------------------------------------------------
            # REAL-TIME RISK CHECK
            # ------------------------------------------------

            mll_loss_threshold = (
                xfa_mll
                - old_balance
            )

            dll_loss_threshold = (
                -XFA_DLL
            )

            # Same real-time/realized net-P&L safeguard used
            # in the Trading Combine.
            day_worst_net_pnl = min(
                net_intraday_mae,
                pnl
            )

            hit_mll = (
                day_worst_net_pnl
                <= mll_loss_threshold
            )

            hit_dll = (
                day_worst_net_pnl
                <= dll_loss_threshold
            )

            if hit_mll and (
                not hit_dll
                or mll_loss_threshold
                >= dll_loss_threshold
            ):

                xfa_failures += 1
                accounts_blown += 1

                account_events.append({
                    "simulation": sim_index + 1,
                    "combine_id": combine_id,
                    "xfa_id": xfa_id,
                    "event": "XFA_FAIL_MLL",
                    "date": date,
                    "reason": "Maximum Loss Limit",
                    "balance": xfa_mll,
                    "mll": xfa_mll,
                    "payouts_on_xfa":
                        xfa_payouts_this_account,
                    "reactivations_used":
                        xfa_reactivations_used
                })

                mll_audit_events.append({
                    "simulation": sim_index + 1,
                    "combine_id": combine_id,
                    "xfa_id": xfa_id,
                    "date": date,
                    "phase": "XFA",
                    "event": "XFA_FAIL_MLL",
                    "balance_before": old_balance,
                    "balance_after": xfa_mll,
                    "mll_before": old_mll,
                    "mll_after": old_mll
                })

                # Back2Funded only if XFA is lost BEFORE
                # its first payout, max 2 reactivations.
                eligible_b2f = (
                    USE_BACK2FUNDED
                    and xfa_payouts_this_account == 0
                    and xfa_reactivations_used
                    < BACK2FUNDED_MAX_REACTIVATIONS
                )

                if eligible_b2f:

                    reactivate_xfa(
                        date
                    )

                else:

                    purchase_new_combine(
                        "XFA MLL breach",
                        date
                    )

                continue

            if hit_dll:

                # DLL = forced break, not failed XFA.
                dll_day_pnl = (
                    -XFA_DLL
                    - ENTRY_SIDE_COST
                )

                xfa_balance = (
                    old_balance
                    + dll_day_pnl
                )

                xfa_cycle_profit += (
                    dll_day_pnl
                )

                xfa_cycle_trading_days += 1

                if dll_day_pnl > xfa_cycle_best_day:
                    xfa_cycle_best_day = dll_day_pnl

                # EOD XFA MLL trails upward but LOCKS at $0.
                candidate_mll = (
                    xfa_balance
                    - XFA_MLL_DISTANCE
                )

                xfa_mll = max(
                    xfa_mll,
                    min(
                        0.0,
                        candidate_mll
                    )
                )

                account_events.append({
                    "simulation": sim_index + 1,
                    "combine_id": combine_id,
                    "xfa_id": xfa_id,
                    "event": "XFA_DLL_LOCKOUT",
                    "date": date,
                    "reason": "Daily Loss Limit",
                    "balance": xfa_balance,
                    "mll": xfa_mll
                })

                mll_audit_events.append({
                    "simulation": sim_index + 1,
                    "combine_id": combine_id,
                    "xfa_id": xfa_id,
                    "date": date,
                    "phase": "XFA",
                    "event": "XFA_DLL_LOCKOUT",
                    "balance_before": old_balance,
                    "balance_after": xfa_balance,
                    "mll_before": old_mll,
                    "mll_after": xfa_mll
                })

                continue

            # ------------------------------------------------
            # ACCEPT NORMAL DAY
            # ------------------------------------------------

            xfa_balance = (
                old_balance
                + pnl
            )

            xfa_cycle_profit += pnl
            xfa_cycle_trading_days += 1

            if pnl > xfa_cycle_best_day:
                xfa_cycle_best_day = pnl

            # EOD XFA MLL trails but never exceeds $0.
            candidate_mll = (
                xfa_balance
                - XFA_MLL_DISTANCE
            )

            xfa_mll = max(
                xfa_mll,
                min(
                    0.0,
                    candidate_mll
                )
            )

            mll_audit_events.append({
                "simulation": sim_index + 1,
                "combine_id": combine_id,
                "xfa_id": xfa_id,
                "date": date,
                "phase": "XFA",
                "event": "TRADING_DAY",
                "balance_before": old_balance,
                "balance_after": xfa_balance,
                "mll_before": old_mll,
                "mll_after": xfa_mll
            })

            # ------------------------------------------------
            # OPTION 2 / CONSISTENCY PAYOUT ELIGIBILITY
            # ------------------------------------------------

            if (
                xfa_cycle_trading_days
                >= XFA_CONSISTENCY_MIN_TRADING_DAYS
                and xfa_cycle_profit > 0
            ):

                consistency_ratio = (
                    max(0.0, xfa_cycle_best_day)
                    / xfa_cycle_profit
                )

                if (
                    consistency_ratio
                    <= XFA_CONSISTENCY_MAX_BEST_DAY_SHARE
                ):

                    eligible_amount = max(
                        0.0,
                        xfa_balance * 0.50
                    )

                    gross_payout = min(
                        eligible_amount,
                        PAYOUT_CAP
                    )

                    if (
                        gross_payout
                        >= MIN_PAYOUT_REQUEST
                    ):

                        trading_days_before_payout = (
                            xfa_cycle_trading_days
                        )

                        best_day_before_payout = (
                            xfa_cycle_best_day
                        )

                        cycle_profit_before_payout = (
                            xfa_cycle_profit
                        )

                        consistency_before_payout = (
                            consistency_ratio
                        )

                        balance_before_payout = (
                            xfa_balance
                        )

                        mll_before_payout = (
                            xfa_mll
                        )

                        trader_payout = (
                            gross_payout
                            * TRADER_SPLIT
                        )

                        payout_total += (
                            trader_payout
                        )

                        payouts += 1
                        xfa_payouts_this_account += 1

                        xfa_balance -= (
                            gross_payout
                        )

                        # After every Option 2 payout, Topstep
                        # resets the consistency calculation and
                        # 3-trading-day count. MLL locks at $0.
                        xfa_mll = 0.0
                        xfa_cycle_trading_days = 0
                        xfa_cycle_profit = 0.0
                        xfa_cycle_best_day = 0.0

                        account_events.append({
                            "simulation":
                                sim_index + 1,
                            "combine_id":
                                combine_id,
                            "xfa_id":
                                xfa_id,
                            "event":
                                "PAYOUT",
                            "date":
                                date,
                            "phase":
                                "XFA",
                            "trading_days_before_payout":
                                trading_days_before_payout,
                            "best_day_before_payout":
                                best_day_before_payout,
                            "cycle_profit_before_payout":
                                cycle_profit_before_payout,
                            "consistency_before_payout":
                                consistency_before_payout,
                            "balance_before_payout":
                                balance_before_payout,
                            "gross_payout":
                                gross_payout,
                            "trader_payout":
                                trader_payout,
                            "balance_after_payout":
                                xfa_balance,
                            "mll_after_payout":
                                xfa_mll,
                            "xfa_payout_number":
                                xfa_payouts_this_account
                        })

                        mll_audit_events.append({
                            "simulation":
                                sim_index + 1,
                            "combine_id":
                                combine_id,
                            "xfa_id":
                                xfa_id,
                            "date":
                                date,
                            "phase":
                                "XFA",
                            "event":
                                "PAYOUT",
                            "balance_before":
                                balance_before_payout,
                            "balance_after":
                                xfa_balance,
                            "mll_before":
                                mll_before_payout,
                            "mll_after":
                                xfa_mll
                        })

    # ========================================================
    # END-OF-SIMULATION ACCOUNTING
    # ========================================================

    # Expire credits through final simulated date for clean
    # end-state reporting.
    expire_reset_credits(
        dates[-1]
    )

    actual_net_pnl = (
        payout_total
        - account_costs
    )

    # Exact cash-cost reconciliation.
    expected_costs = (
        combine_subscriptions_purchased
        * COMBINE_MONTHLY_COST
        + monthly_rebills
        * COMBINE_MONTHLY_COST
        + paid_resets
        * PAID_RESET_COST
        + back2funded_reactivations
        * BACK2FUNDED_COST
    )

    if abs(
        account_costs
        - expected_costs
    ) > 0.01:

        raise ValueError(
            f"Cost reconciliation failed "
            f"in simulation {sim_index + 1}: "
            f"{account_costs} vs {expected_costs}"
        )

    if (
        reset_credits_used
        + reset_credits_expired
        + len(reset_credit_dates)
        != reset_credits_earned
    ):

        raise ValueError(
            f"Reset Credit reconciliation failed "
            f"in simulation {sim_index + 1}."
        )

    if (
        reset_credits_used
        > reset_credits_earned
    ):

        raise ValueError(
            f"Used more Reset Credits than earned "
            f"in simulation {sim_index + 1}."
        )

    result = {
        "simulation": sim_index + 1,
        "trading_days": len(daily_pnl),

        "combine_subscriptions_purchased":
            combine_subscriptions_purchased,

        "monthly_rebills":
            monthly_rebills,

        "paid_resets":
            paid_resets,

        "reset_credits_earned":
            reset_credits_earned,

        "reset_credits_used":
            reset_credits_used,

        "reset_credits_expired":
            reset_credits_expired,

        "reset_credits_remaining":
            len(reset_credit_dates),

        "combine_passes":
            combine_passes,

        "xfa_accounts_earned":
            xfa_accounts_earned,

        "back2funded_reactivations":
            back2funded_reactivations,

        "xfa_failures":
            xfa_failures,

        "accounts_blown":
            accounts_blown,

        "payouts":
            payouts,

        "trader_payouts":
            payout_total,

        "account_costs":
            account_costs,

        "actual_net_pnl":
            actual_net_pnl
    }

    return (
        result,
        account_events,
        mll_audit_events,
        billing_events
    )


# ============================================================
# RUN ALL 10,000 SIMULATIONS
# ============================================================

print()
print("==============================")
print("RUNNING TOPSTEP OPTION 2 — 10:00 / 100 TICKS / 8 MNQ")
print("==============================")

results = []
all_events = []
all_mll_audit_events = []
all_billing_events = []

for sim_index in range(
    NUM_SIMULATIONS
):

    (
        result,
        events,
        mll_events,
        billing_events
    ) = run_simulation(
        sim_index
    )

    results.append(
        result
    )

    all_events.extend(
        events
    )

    all_mll_audit_events.extend(
        mll_events
    )

    all_billing_events.extend(
        billing_events
    )

    if (
        sim_index + 1
    ) % 1000 == 0:

        print(
            f"Completed "
            f"{sim_index + 1:,} / "
            f"{NUM_SIMULATIONS:,}"
        )


# ============================================================
# DATAFRAMES
# ============================================================

results_df = pd.DataFrame(
    results
)

events_df = pd.DataFrame(
    all_events
)

mll_audit_df = pd.DataFrame(
    all_mll_audit_events
)

billing_audit_df = pd.DataFrame(
    all_billing_events
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

events_df.to_csv(
    ACCOUNT_LOG_FILE,
    index=False
)

mll_audit_df.to_csv(
    MLL_AUDIT_FILE,
    index=False
)

billing_audit_df.to_csv(
    BILLING_AUDIT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("==============================")
print("TOPSTEP OPTION 2 — 10:00 / 100 TICKS / 8 MNQ RESULTS")
print("==============================")

print(
    f"Simulations: "
    f"{len(results_df):,}"
)

print(
    f"Mean Combine subscriptions purchased: "
    f"{results_df.combine_subscriptions_purchased.mean():.2f}"
)

print(
    f"Mean monthly rebills: "
    f"{results_df.monthly_rebills.mean():.2f}"
)

print(
    f"Mean paid Resets: "
    f"{results_df.paid_resets.mean():.2f}"
)

print(
    f"Mean Reset Credits earned: "
    f"{results_df.reset_credits_earned.mean():.2f}"
)

print(
    f"Mean Reset Credits used: "
    f"{results_df.reset_credits_used.mean():.2f}"
)

print(
    f"Mean Reset Credits expired: "
    f"{results_df.reset_credits_expired.mean():.2f}"
)

print(
    f"Mean Combine passes: "
    f"{results_df.combine_passes.mean():.2f}"
)

print(
    f"Mean XFA accounts earned: "
    f"{results_df.xfa_accounts_earned.mean():.2f}"
)

print(
    f"Mean Back2Funded reactivations: "
    f"{results_df.back2funded_reactivations.mean():.2f}"
)

print(
    f"Mean XFA failures: "
    f"{results_df.xfa_failures.mean():.2f}"
)

print(
    f"Mean payouts: "
    f"{results_df.payouts.mean():.2f}"
)

print(
    f"Mean trader payouts: "
    f"${results_df.trader_payouts.mean():,.2f}"
)

print(
    f"Median trader payouts: "
    f"${results_df.trader_payouts.median():,.2f}"
)

print(
    f"Mean account costs: "
    f"${results_df.account_costs.mean():,.2f}"
)

print(
    f"TOTAL trader payouts across all simulations: "
    f"${results_df.trader_payouts.sum():,.2f}"
)

print(
    f"TOTAL account costs across all simulations: "
    f"${results_df.account_costs.sum():,.2f}"
)

print(
    f"TOTAL ACTUAL net P&L across all simulations: "
    f"${results_df.actual_net_pnl.sum():,.2f}"
)

print(
    f"Mean ACTUAL net P&L: "
    f"${results_df.actual_net_pnl.mean():,.2f}"
)

print(
    f"Median ACTUAL net P&L: "
    f"${results_df.actual_net_pnl.median():,.2f}"
)

print(
    f"Std. dev. ACTUAL net P&L: "
    f"${results_df.actual_net_pnl.std():,.2f}"
)

print(
    f"5th percentile ACTUAL net P&L: "
    f"${results_df.actual_net_pnl.quantile(.05):,.2f}"
)

print(
    f"95th percentile ACTUAL net P&L: "
    f"${results_df.actual_net_pnl.quantile(.95):,.2f}"
)

profitable = (
    results_df.actual_net_pnl
    > 0
)

losing = (
    results_df.actual_net_pnl
    < 0
)

zero = (
    results_df.actual_net_pnl
    == 0
)

print(
    f"Profitable simulations: "
    f"{profitable.sum():,} "
    f"({profitable.mean() * 100:.2f}%)"
)

print(
    f"Losing simulations: "
    f"{losing.sum():,} "
    f"({losing.mean() * 100:.2f}%)"
)

print(
    f"Exactly zero: "
    f"{zero.sum():,}"
)

print()
print("==============================")
print("MODEL CONSTANTS")
print("==============================")

print(
    f"Combine monthly cost: "
    f"${COMBINE_MONTHLY_COST:,.2f}"
)

print(
    f"Paid Reset cost: "
    f"${PAID_RESET_COST:,.2f}"
)

print(
    f"Back2Funded cost: "
    f"${BACK2FUNDED_COST:,.2f}"
)

print(
    f"Entry time: "
    f"{ENTRY_TIME_REQUIRED}"
)

print(
    f"TP/SL distance: "
    f"{TICKS} ticks ({TP_SL_POINTS:.2f} points)"
)

print(
    f"Contracts: "
    f"{CONTRACTS} MNQ"
)

print(
    f"Gross TP/SL: "
    f"${GROSS_TP_SL_DOLLARS:,.2f}"
)

print(
    f"MNQ RT cost / contract: "
    f"${MNQ_RT_COST_PER_CONTRACT:,.2f}"
)

print(
    f"{CONTRACTS}-MNQ RT cost / trade: "
    f"${ROUND_TURN_COST:,.2f}"
)

print(
    f"XFA Consistency payout cap: "
    f"${PAYOUT_CAP:,.2f}"
)

print(
    f"Option 2 minimum trading days: "
    f"{XFA_CONSISTENCY_MIN_TRADING_DAYS}"
)

print(
    f"Option 2 max best-day share: "
    f"{XFA_CONSISTENCY_MAX_BEST_DAY_SHARE * 100:.0f}%"
)

print(
    f"Back2Funded enabled: "
    f"{USE_BACK2FUNDED}"
)

print(
    f"Trader split: "
    f"{TRADER_SPLIT * 100:.0f}%"
)

print()
print("Saved:")
print(OUTPUT_FILE)
print(ACCOUNT_LOG_FILE)
print(MLL_AUDIT_FILE)
print(BILLING_AUDIT_FILE)


# ============================================================
# FIVE-ACCOUNT PORTFOLIO + RAW/FIVE-YEAR ROBUSTNESS ANALYSIS
# ============================================================
#
# IMPORTANT:
# This section DOES NOT change any of the audited one-account
# 50K Topstep Option 2 rules above.
#
# It models five identical 50K accounts traded simultaneously
# with the same copied 10:00 / 100-tick / 8-MNQ trade.
#
# Because all five accounts:
#   - start together,
#   - receive the exact same copied trades,
#   - use identical Topstep rules,
# their state paths are identical. Therefore the exact
# five-account portfolio cash flows/counts are 5x the
# one-account lifecycle, while profitability percentages are
# unchanged.
#
# Outputs:
#   1) raw strategy — full sample
#   2) raw strategy — exact five-year split
#   3) Topstep Option 2 — five-account full sample
#   4) Topstep Option 2 — five-account exact five-year split
# ============================================================

ACTIVE_ACCOUNTS = 5

FIRST_START = pd.Timestamp("2016-08-08")
FIRST_END = pd.Timestamp("2021-08-06")
SECOND_START = pd.Timestamp("2021-08-09")
SECOND_END = pd.Timestamp("2026-08-05")

# Preserve the exact full-sample state generated above.
FULL_PATHS = paths.copy()
FULL_CLEAN_DAYS = clean_days.copy()
FULL_DATES = dates.copy()
FULL_LONG_NET_PNL = long_net_pnl.copy()
FULL_SHORT_NET_PNL = short_net_pnl.copy()
FULL_LONG_MAE = long_mae.copy()
FULL_SHORT_MAE = short_mae.copy()

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def raw_summary_for_mask(label, mask):
    idx = np.where(mask)[0]
    p = FULL_PATHS[:, idx]
    d = FULL_CLEAN_DAYS.iloc[idx].reset_index(drop=True)

    lp = d["long_pnl"].to_numpy(dtype=float)
    sp = d["short_pnl"].to_numpy(dtype=float)

    # Gross raw P&L, matching the underlying strategy analysis.
    chosen = np.where(p == 1, lp[None, :], sp[None, :])

    long_result_arr = d["long_result"].astype(str).to_numpy()
    short_result_arr = d["short_result"].astype(str).to_numpy()

    chosen_result = np.where(
        p == 1,
        long_result_arr[None, :],
        short_result_arr[None, :]
    )

    wins = int(np.sum(chosen_result == "WIN"))
    losses = int(np.sum(chosen_result == "LOSS"))
    time_exits = int(np.sum(chosen_result == "TIME_EXIT"))
    total_obs = chosen_result.size
    resolved = wins + losses

    # Average number of each outcome in one randomized path.
    avg_wins = wins / NUM_SIMULATIONS
    avg_losses = losses / NUM_SIMULATIONS
    avg_time_exits = time_exits / NUM_SIMULATIONS

    mean_path_pnl = chosen.sum(axis=1).mean()
    median_path_pnl = np.median(chosen.sum(axis=1))
    mean_pnl_trade = chosen.mean()

    tp_sl_win_rate = (
        wins / resolved * 100.0
        if resolved else np.nan
    )

    return {
        "Period": label,
        "Start": d["date_dt"].min().date(),
        "End": d["date_dt"].max().date(),
        "Trading days per path": len(d),
        "Mean wins per path": avg_wins,
        "Mean losses per path": avg_losses,
        "Mean time exits per path": avg_time_exits,
        "TP/SL win rate %": tp_sl_win_rate,
        "Mean raw P&L per trade": mean_pnl_trade,
        "Mean raw total P&L per path": mean_path_pnl,
        "Median raw total P&L per path": median_path_pnl,
        "Total randomized trade observations": total_obs,
    }


def set_engine_period(idx):
    """
    Point the already-audited run_simulation() function at one
    exact date slice without changing its Topstep rules.
    """
    global paths, clean_days, dates
    global long_net_pnl, short_net_pnl
    global long_mae, short_mae

    paths = FULL_PATHS[:, idx]
    clean_days = (
        FULL_CLEAN_DAYS
        .iloc[idx]
        .copy()
        .reset_index(drop=True)
    )
    dates = FULL_DATES[idx]
    long_net_pnl = FULL_LONG_NET_PNL[idx]
    short_net_pnl = FULL_SHORT_NET_PNL[idx]
    long_mae = FULL_LONG_MAE[idx]
    short_mae = FULL_SHORT_MAE[idx]


def run_period_topstep(label, idx):
    set_engine_period(idx)

    period_results = []

    print()
    print("==============================")
    print(f"RUNNING TOPSTEP SPLIT: {label}")
    print("==============================")
    print(f"Trading days: {len(idx):,}")

    for sim_index in range(NUM_SIMULATIONS):
        result, _, _, _ = run_simulation(sim_index)
        period_results.append(result)

        if (sim_index + 1) % 1000 == 0:
            print(
                f"Completed {sim_index + 1:,} / "
                f"{NUM_SIMULATIONS:,}"
            )

    df = pd.DataFrame(period_results)
    return df


def five_account_portfolio(df):
    """
    Convert the exact one-account result into five simultaneous
    copied 50K accounts.

    Every account receives the same trade and has identical
    rules/state, so all account counts and real cash flows are
    multiplied by five.
    """
    out = df.copy()

    count_cols = [
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
    ]

    money_cols = [
        "trader_payouts",
        "account_costs",
        "actual_net_pnl",
    ]

    for c in count_cols + money_cols:
        if c in out.columns:
            out[c] = out[c] * ACTIVE_ACCOUNTS

    out["active_parallel_accounts"] = ACTIVE_ACCOUNTS
    return out


def topstep_summary(label, df):
    pnl = df["actual_net_pnl"]
    return {
        "Period": label,
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


# ------------------------------------------------------------
# 1) RAW STRATEGY — FULL SAMPLE + FIVE-YEAR SPLIT
# ------------------------------------------------------------

full_mask = np.ones(
    len(FULL_CLEAN_DAYS),
    dtype=bool
)

first_mask = (
    (FULL_CLEAN_DAYS["date_dt"] >= FIRST_START)
    & (FULL_CLEAN_DAYS["date_dt"] <= FIRST_END)
).to_numpy()

second_mask = (
    (FULL_CLEAN_DAYS["date_dt"] >= SECOND_START)
    & (FULL_CLEAN_DAYS["date_dt"] <= SECOND_END)
).to_numpy()

raw_full = raw_summary_for_mask(
    "Full sample",
    full_mask
)

raw_first = raw_summary_for_mask(
    "2016-08-08 to 2021-08-06",
    first_mask
)

raw_second = raw_summary_for_mask(
    "2021-08-09 to 2026-08-05",
    second_mask
)

raw_summary_df = pd.DataFrame(
    [raw_full, raw_first, raw_second]
)

raw_summary_df.to_csv(
    "raw_10am_100tick_8mnq_full_and_5year_split.csv",
    index=False
)

print()
print("==============================")
print("RAW STRATEGY — FULL + 5-YEAR SPLIT")
print("==============================")
print(raw_summary_df.to_string(index=False))


# ------------------------------------------------------------
# 2) FIVE 50K ACCOUNTS — FULL SAMPLE
# ------------------------------------------------------------
#
# results_df was generated above by the unchanged audited
# one-account full-sample engine.
# ------------------------------------------------------------

five_full_df = five_account_portfolio(
    results_df
)

five_full_df.to_csv(
    "topstep_option2_5x50k_10am_full_results.csv",
    index=False
)

full_portfolio_summary = topstep_summary(
    "Full sample",
    five_full_df
)


# ------------------------------------------------------------
# 3) FIVE 50K ACCOUNTS — EXACT FIVE-YEAR SPLIT
# ------------------------------------------------------------

first_idx = np.where(first_mask)[0]
second_idx = np.where(second_mask)[0]

first_one_df = run_period_topstep(
    "2016-08-08 to 2021-08-06",
    first_idx
)

second_one_df = run_period_topstep(
    "2021-08-09 to 2026-08-05",
    second_idx
)

five_first_df = five_account_portfolio(
    first_one_df
)

five_second_df = five_account_portfolio(
    second_one_df
)

five_first_df.to_csv(
    "topstep_option2_5x50k_10am_first5_results.csv",
    index=False
)

five_second_df.to_csv(
    "topstep_option2_5x50k_10am_second5_results.csv",
    index=False
)

first_summary = topstep_summary(
    "2016-08-08 to 2021-08-06",
    five_first_df
)

second_summary = topstep_summary(
    "2021-08-09 to 2026-08-05",
    five_second_df
)

topstep_summary_df = pd.DataFrame([
    full_portfolio_summary,
    first_summary,
    second_summary,
])

topstep_summary_df.to_csv(
    "topstep_option2_5x50k_10am_full_and_5year_summary.csv",
    index=False
)

print()
print("==============================")
print("TOPSTEP OPTION 2 — FIVE 50K ACCOUNTS")
print("==============================")
print(topstep_summary_df.to_string(index=False))

print()
print("==============================")
print("MODEL SETTINGS")
print("==============================")
print("Parallel accounts: 5")
print("Account size: $50,000 each")
print("Entry: 10:00 ET")
print("Direction: randomized LONG/SHORT")
print("TP/SL: 100 ticks = 25 points")
print("Contracts: 8 MNQ PER ACCOUNT")
print("Gross TP/SL: +/- $400 PER ACCOUNT")
print("Copied gross TP/SL across 5 accounts: +/- $2,000")
print(f"RT cost per account/trade: ${ROUND_TURN_COST:,.2f}")
print(f"RT cost across 5 copied accounts/trade: ${ROUND_TURN_COST * ACTIVE_ACCOUNTS:,.2f}")
print("Topstep rules: unchanged audited 50K Option 2 rules")
print("Back2Funded: OFF")

print()
print("Saved:")
print("raw_10am_100tick_8mnq_full_and_5year_split.csv")
print("topstep_option2_5x50k_10am_full_results.csv")
print("topstep_option2_5x50k_10am_first5_results.csv")
print("topstep_option2_5x50k_10am_second5_results.csv")
print("topstep_option2_5x50k_10am_full_and_5year_summary.csv")
