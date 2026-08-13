import pandas as pd

FIRST_FILE = "topstep_option2_5x50k_10am_first5_results.csv"
SECOND_FILE = "topstep_option2_5x50k_10am_second5_results.csv"
OUTPUT_FILE = "topstep_5year_exhibit3_net_pnl.csv"

first = pd.read_csv(FIRST_FILE)
second = pd.read_csv(SECOND_FILE)

required = ["simulation", "actual_net_pnl"]

for name, df in [("first", first), ("second", second)]:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} file missing columns: {missing}")

if len(first) != 10_000 or len(second) != 10_000:
    raise ValueError(
        f"Expected 10,000 rows in each file. "
        f"Found {len(first):,} and {len(second):,}."
    )

out = pd.DataFrame({
    "simulation": first["simulation"],
    "first5_net_pnl": first["actual_net_pnl"],
    "second5_net_pnl": second["actual_net_pnl"]
})

out.to_csv(OUTPUT_FILE, index=False)

print(f"Saved: {OUTPUT_FILE}")
print()
print("FIRST FIVE YEARS")
print(f"Mean: ${out['first5_net_pnl'].mean():,.2f}")
print(f"Median: ${out['first5_net_pnl'].median():,.2f}")
print(
    f"Profitable: {(out['first5_net_pnl'] > 0).mean() * 100:.2f}%"
)

print()
print("SECOND FIVE YEARS")
print(f"Mean: ${out['second5_net_pnl'].mean():,.2f}")
print(f"Median: ${out['second5_net_pnl'].median():,.2f}")
print(
    f"Profitable: {(out['second5_net_pnl'] > 0).mean() * 100:.2f}%"
)
