from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# RAW STRATEGY — COMPLETE
# 10:00 ET / 100-TICK TP-SL / 8 MNQ
# 10,000 RANDOM LONG/SHORT PATHS
#
# Produces:
#   1) Full-sample raw strategy results
#   2) Exact five-year split raw strategy results
#
# This is the standalone RAW portion of the final complete
# 10:00 / 100-tick / 8-MNQ simulation.
# ============================================================

NQ_FILE = "NQ_data.csv"

DAY_FILE = "mnq_10am_100tick_8con_clean_days.csv"
PATH_FILE = "mnq_10am_100tick_8con_10000_paths.npy"
OUTPUT_FILE = "raw_10am_100tick_8mnq_full_and_5year_split.csv"

NUM_SIMULATIONS = 10_000
BASE_SEED = 42

REQUIRED_START_TIME = "09:30:00"
ENTRY_TIME_REQUIRED = "10:00:00"
REQUIRED_END_TIME = "16:00:00"

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

# Kept here for documentation/consistency with the complete model.
# The raw summary below is GROSS raw strategy P&L, matching the
# final raw output used in the complete simulation.
MNQ_RT_COST_PER_CONTRACT = 1.22
ROUND_TURN_COST = MNQ_RT_COST_PER_CONTRACT * CONTRACTS

FIRST_START = pd.Timestamp("2016-08-08")
FIRST_END = pd.Timestamp("2021-08-06")
SECOND_START = pd.Timestamp("2021-08-09")
SECOND_END = pd.Timestamp("2026-08-05")


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
    names=["date", "time", "O", "H", "L", "C", "volume"]
)

for col in ["O", "H", "L", "C", "volume"]:
    nq_full[col] = pd.to_numeric(nq_full[col], errors="coerce")

nq_full["datetime"] = pd.to_datetime(
    nq_full["date"].astype(str)
    + " "
    + nq_full["time"].astype(str),
    errors="coerce"
)

nq_full = nq_full.dropna(
    subset=["datetime", "O", "H", "L", "C"]
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

for date_key, day in nq_full.groupby(nq_full["datetime"].dt.date):

    day = day.sort_values("datetime").reset_index(drop=True)

    first_time = day.iloc[0]["datetime"].strftime("%H:%M:%S")
    last_time = day.iloc[-1]["datetime"].strftime("%H:%M:%S")

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

    post_entry = day[day["datetime"] > entry_time]

    # -------------------------
    # LONG
    # -------------------------

    long_result = None
    long_exit_price = None
    long_exit_time = None

    for _, bar in post_entry.iterrows():

        hit_target = bar["H"] >= long_target
        hit_stop = bar["L"] <= long_stop

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
        final_close = float(day.iloc[-1]["C"])
        long_result = "TIME_EXIT"
        long_exit_price = final_close
        long_exit_time = day.iloc[-1]["datetime"]
        long_pnl = (
            (final_close - entry)
            * MNQ_DOLLARS_PER_POINT
            * CONTRACTS
        )

    elif long_result == "WIN":
        long_pnl = GROSS_TP_SL_DOLLARS

    elif long_result == "LOSS":
        long_pnl = -GROSS_TP_SL_DOLLARS

    else:
        long_pnl = np.nan

    # -------------------------
    # SHORT
    # -------------------------

    short_result = None
    short_exit_price = None
    short_exit_time = None

    for _, bar in post_entry.iterrows():

        hit_target = bar["L"] <= short_target
        hit_stop = bar["H"] >= short_stop

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
        final_close = float(day.iloc[-1]["C"])
        short_result = "TIME_EXIT"
        short_exit_price = final_close
        short_exit_time = day.iloc[-1]["datetime"]
        short_pnl = (
            (entry - final_close)
            * MNQ_DOLLARS_PER_POINT
            * CONTRACTS
        )

    elif short_result == "WIN":
        short_pnl = GROSS_TP_SL_DOLLARS

    elif short_result == "LOSS":
        short_pnl = -GROSS_TP_SL_DOLLARS

    else:
        short_pnl = np.nan

    if (
        long_result == "AMBIGUOUS"
        or short_result == "AMBIGUOUS"
    ):
        ambiguous_days.append(str(date_key))

    day_results.append({
        "date": str(date_key),
        "entry_time": entry_time,
        "entry": entry,

        "long_pnl": long_pnl,
        "short_pnl": short_pnl,

        "long_result": long_result,
        "short_result": short_result,

        "long_exit_price": long_exit_price,
        "short_exit_price": short_exit_price,

        "long_exit_time": long_exit_time,
        "short_exit_time": short_exit_time
    })


day_results = pd.DataFrame(day_results)

clean_days = day_results[
    day_results["long_pnl"].notna()
    & day_results["short_pnl"].notna()
].copy().reset_index(drop=True)

clean_days.to_csv(DAY_FILE, index=False)

print(f"Full-session exclusions: {excluded_nonfull:,}")
print(f"Full sessions missing 10:00 bar: {excluded_no_10am:,}")
print(f"Ambiguous days excluded: {len(ambiguous_days):,}")
print(f"Final clean trading days: {len(clean_days):,}")
print(f"Gross TP/SL value: ${GROSS_TP_SL_DOLLARS:,.2f}")
print(f"8-MNQ round-turn commission: ${ROUND_TURN_COST:,.2f}")


# ============================================================
# GENERATE EXACT SAME 10,000 RANDOM LONG/SHORT PATHS
# ============================================================

rng = np.random.default_rng(BASE_SEED)

paths = rng.integers(
    0,
    2,
    size=(NUM_SIMULATIONS, len(clean_days)),
    dtype=np.int8
)

np.save(PATH_FILE, paths)

print()
print(f"Random path shape: {paths.shape}")
print(f"Clean trading days: {len(clean_days):,}")

if paths.shape[0] != NUM_SIMULATIONS:
    raise ValueError(
        f"Expected {NUM_SIMULATIONS:,} simulations "
        f"but found {paths.shape[0]:,}."
    )

if paths.shape[1] != len(clean_days):
    raise ValueError(
        "Random-path columns do not match clean trading days."
    )

clean_days["date_dt"] = pd.to_datetime(
    clean_days["date"]
).dt.normalize()


# ============================================================
# RAW SUMMARY HELPER
# ============================================================

def raw_summary_for_mask(label, mask):

    idx = np.where(mask)[0]
    p = paths[:, idx]
    d = clean_days.iloc[idx].reset_index(drop=True)

    lp = d["long_pnl"].to_numpy(dtype=float)
    sp = d["short_pnl"].to_numpy(dtype=float)

    # IMPORTANT:
    # Gross raw P&L, matching the underlying strategy analysis
    # in the final COMPLETE simulation.
    chosen = np.where(
        p == 1,
        lp[None, :],
        sp[None, :]
    )

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

    avg_wins = wins / NUM_SIMULATIONS
    avg_losses = losses / NUM_SIMULATIONS
    avg_time_exits = time_exits / NUM_SIMULATIONS

    path_pnl = chosen.sum(axis=1)

    mean_path_pnl = path_pnl.mean()
    median_path_pnl = np.median(path_pnl)
    mean_pnl_trade = chosen.mean()

    tp_sl_win_rate = (
        wins / resolved * 100.0
        if resolved
        else np.nan
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


# ============================================================
# FULL SAMPLE + EXACT FIVE-YEAR SPLIT
# ============================================================

full_mask = np.ones(
    len(clean_days),
    dtype=bool
)

first_mask = (
    (clean_days["date_dt"] >= FIRST_START)
    & (clean_days["date_dt"] <= FIRST_END)
).to_numpy()

second_mask = (
    (clean_days["date_dt"] >= SECOND_START)
    & (clean_days["date_dt"] <= SECOND_END)
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
    OUTPUT_FILE,
    index=False
)

print()
print("==============================")
print("RAW STRATEGY — FULL + 5-YEAR SPLIT")
print("==============================")
print(raw_summary_df.to_string(index=False))

print()
print("Saved:")
print(DAY_FILE)
print(PATH_FILE)
print(OUTPUT_FILE)
