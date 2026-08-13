import pandas as pd
from pathlib import Path

INPUT_FILE = "topstep_option2_5x50k_10am_full_results.csv"
OUTPUT_FILE = "topstep_5x50k_10000_path_net_pnl.csv"

if not Path(INPUT_FILE).exists():
    raise FileNotFoundError(
        f"{INPUT_FILE} not found. Run the final Topstep COMPLETE simulation first."
    )

df = pd.read_csv(INPUT_FILE)

required = ["simulation", "actual_net_pnl"]
missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

out = df[["simulation", "actual_net_pnl"]].copy()

if len(out) != 10_000:
    raise ValueError(
        f"Expected 10,000 simulations, found {len(out):,}"
    )

out.to_csv(OUTPUT_FILE, index=False)

print(f"Saved: {OUTPUT_FILE}")
print(f"Rows: {len(out):,}")
print(f"Mean net P&L: ${out['actual_net_pnl'].mean():,.2f}")
print(f"Median net P&L: ${out['actual_net_pnl'].median():,.2f}")
print(
    f"Profitable simulations: {(out['actual_net_pnl'] > 0).sum():,} "
    f"({(out['actual_net_pnl'] > 0).mean() * 100:.2f}%)"
)
