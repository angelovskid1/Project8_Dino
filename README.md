# Project 8: Moving Average Backtest Extension

## Overview
This project extends a Python backtesting script to evaluate 10 different moving average crossover strategies (EMA and SMA) on a watchlist of stocks. The goal is to compare performance across different timeframes and provide detailed execution logs for analysis.

## Features
-   **10 Strategies Tested**:
    -   EMA: 9/20, 9/50, 20/50, 30/60, 50/200
    -   SMA: 9/20, 9/50, 20/50, 30/60, 50/200
-   **Global Trade Logging**: innovative logging system that records every buy/sell event to `trade_log.csv`.
-   **Comprehensive Summary**: Generates a side-by-side comparison of final portfolio values for all strategies.
-   **Visualization**: Equity curves for key strategies.

## Setup

1.  Ensure you have Python 3.x installed.
2.  Install the required dependencies:
    ```bash
    pip install pandas yfinance matplotlib python-docx numpy
    ```

## Usage

1.  Update `watchlist_leo_project7.csv` with your desired stock symbols (optional).
2.  Run the script:
    ```bash
    python leo_backtest.py
    ```

## Outputs
All results are saved in the `outputs/` directory:

-   **`backtest_summary.csv`**: Table of Final Portfolio Values for each symbol and strategy.
-   **`trade_log.csv`**: Detailed log of all executed trades (Symbol, Strategy, Date, Action, Price, Shares).
-   **`backtest_report.docx`**: Word document summarizing the results.
-   **`*_equity.png`**: Equity curve plots for selected strategies.

## strategies
-   **Buy Signal**: Short MA crosses *above* Long MA.
-   **Sell Signal**: Short MA crosses *below* Long MA.
-   **Initial Capital**: $10,000 per strategy.
