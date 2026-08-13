# The Inherent Edge of Retail Proprietary Trading Firms

This repository contains the code, simulation outputs, audit materials, and exhibit data associated with the paper **"The Inherent Edge of Retail Proprietary Trading Firms."**

The study examines retail proprietary trading accounts as path-dependent financial contracts. A randomized trading strategy is used as a neutral baseline, and the same randomized trading paths are subsequently evaluated under a simulated proprietary trading framework. This structure allows the economic effects of the prop-firm rules to be separated from the underlying raw trading performance.

## Repository Structure

### Raw Strategy

- `raw_backtest_code.py`  
  Main raw-strategy simulation code. Constructs the eligible trading-day sample, evaluates the trading strategy, generates randomized paths, and calculates raw strategy results.

- `raw_backtest_results.csv`  
  Summary results from the raw-strategy backtest, including full-sample and five-year subperiod results.

- `raw_clean_days.csv`  
  Cleaned day-level trading data used to construct the randomized paths.

- `raw_randomized_paths.npy`  
  NumPy array containing the 10,000 randomized trading paths used throughout the Monte Carlo analysis.

### Prop-Firm Simulation

- `topstep_backtest_code.py`  
  Main prop-firm simulation code. Applies the modeled Topstep account rules to the randomized trading paths and evaluates full-sample and five-year subperiod outcomes.

- `topstep_backtest_results.csv`  
  Summary output from the prop-firm simulation.

- `topstep_full_results.csv`  
  Full-sample path-level prop-firm simulation results.

- `topstep_first5_results.csv`  
  Path-level prop-firm results for the first five-year subperiod.

- `topstep_second5_results.csv`  
  Path-level prop-firm results for the second five-year subperiod.

## Paper Exhibits

### Exhibit 1 — Raw Strategy P&L Distribution

- `exhibit1_raw_pnl_results.csv`  
  Data used to construct Exhibit 1, showing the distribution of raw strategy P&L across 10,000 randomized paths.

### Exhibit 2 — Prop-Firm Net P&L Distribution

- `exhibit2_export_code.py`  
  Code used to extract and prepare the path-level net P&L observations for Exhibit 2.

- `exhibit2_topstep_pnl_results.csv`  
  Data used to construct Exhibit 2, showing the distribution of net prop-firm P&L across 10,000 randomized paths.

### Exhibit 3 — Five-Year Subperiod Comparison

- `exhibit3_export_code.py`  
  Code used to construct the five-year comparison dataset for Exhibit 3.

- `exhibit3_five_year_results.csv`  
  Data used to construct Exhibit 3, comparing net prop-firm P&L distributions across the first and second five-year subperiods.

## Audit

- `audit_full_code.py`  
  Independent audit code used to validate the implementation of the simulated prop-firm rules and accounting logic.

- `audit_full_results.txt`  
  Output from the full simulation audit.

## Reproducibility

The analysis follows the same randomized trading paths through both the raw-strategy and prop-firm simulations. This permits direct comparison between the economics of the underlying trading strategy and the outcomes generated after applying the prop-firm account rules.

The `.npy` file contains the randomized path data required by the simulation and can be loaded in Python using NumPy.

## Requirements

The analysis was conducted in Python using standard scientific-computing packages including:

- Python
- pandas
- NumPy

## Paper

The accompanying paper develops the interpretation of retail proprietary trading accounts as path-dependent, barrier-like contracts and evaluates whether the fee structure of these accounts can produce economic value even when the underlying trading strategy has approximately zero raw edge.
