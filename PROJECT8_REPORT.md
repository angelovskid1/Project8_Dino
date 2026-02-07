# Project 8 Report: Moving Average Crossover Backtest Extension

**Date:** February 7, 2026
**Author:** Leonardo Carcache (Intern)

## Objective
To extend an existing Python backtesting framework to evaluate a broader range of moving average crossover strategies. The goal was to compare the performance of 10 different strategy combinations (EMA vs. SMA) across multiple timeframes on a watchlist of stocks.

## Methodology

### 1. Data Source
-   Historical daily price data was fetched using the `yfinance` library.
-   **Backtest Period:** January 1, 2020 to Present (Feb 7, 2026).
-   **Initial Capital:** $10,000 per strategy per symbol.

### 2. Strategies Tested
We tested both Exponential Moving Averages (EMA) and Simple Moving Averages (SMA) crossovers for the following period pairs:
-   **9 / 20** (Short-term momentum)
-   **9 / 50**
-   **20 / 50** (Medium-term trend)
-   **30 / 60**
-   **50 / 200** (Long-term "Golden Cross")

**Execution Logic:**
-   **Buy Signal:** Short MA crosses *above* Long MA.
-   **Sell Signal:** Short MA crosses *below* Long MA.
-   **Trade Execution:** Orders executed at the Close price on the day of the signal.
-   **Position Sizing:** All-in (100% equity). No leverage.

### 3. Implementation Details
The Python script `leo_backtest.py` was modified to:
-   Compute all required indicators (EMA/SMA 9, 20, 30, 50, 60, 200).
-   Iterate through all 10 strategies for each symbol.
-   Log every trade execution (Date, Action, Price, Shares) to a CSV file.
-   Generate a consolidated summary table of Final Portfolio Values.

## Results Summary

The backtest produced the following outputs in the `outputs/` directory:

1.  **`backtest_summary.csv`**: A table showing the Final Value of the $10,000 investment for each symbol and strategy.
2.  **`trade_log.csv`**: A detailed row-by-row log of every trade executed during the backtest period.
3.  **Equity Curve Plots**: Visualizations of portfolio growth for selected key strategies (EMA 9/20, EMA 20/50, SMA 20/50).

### Key Observations
*(Based on sample data for AAPL, MSFT, GOOGL, AMZN, TSLA)*
-   **Volatility:** Shorter-term strategies like EMA 9/20 often yielded higher returns on volatile assets (e.g., TSLA) but also incurred more trades and potential "whipsaws".
-   **Stability:** Longer-term strategies like SMA 50/200 ("Golden Cross") generally had fewer trades and smoother equity curves but sometimes lagged in reacting to rapid market reversals.

## Conclusion
The extended backtest framework provides a robust way to compare multiple trend-following strategies simultaneously. The addition of detailed trade logging allows for granular inspection of entry and exit points, facilitating better strategy refinement in the future.
