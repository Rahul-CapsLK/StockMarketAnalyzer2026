# Stock Market Pattern Analyzer

A Python + Streamlit college major project that analyzes stock-price history and demonstrates three Data Structures and Algorithms (DSA) concepts.

## Features
- Interactive price chart with 5-day and 20-day simple moving averages.
- CSV upload and data validation/cleaning.
- Sliding-window minimum, maximum, and range statistics.
- Heap-based top-K volatile-day ranking.
- Segment Tree range minimum/maximum queries.
- Basic moving-average crossover and closing-price breakout signals.
- Download cleaned/analyzed data as CSV.
- Built-in deterministic demo data, so it runs without an API key or internet data source.

## Requirements
- Python 3.10 or newer recommended.
- Internet is needed only to install packages the first time. The app itself can run locally with the demo dataset.

## Windows setup
1. Extract the ZIP file.
2. Open the extracted `Stock_Market_Pattern_Analyzer` folder.
3. Click the folder address bar, type `cmd`, and press Enter.
4. Create a virtual environment:
   ```
   python -m venv .venv
   ```
5. Activate it:
   ```
   .venv\Scripts\activate
   ```
6. Install dependencies:
   ```
   python -m pip install -r requirements.txt
   ```
7. Start the app:
   ```
   python -m streamlit run app.py
   ```
8. Streamlit will show a local address, usually `http://localhost:8501`, and may open it in your browser.

## CSV format
Minimum required column: `Close`.
Recommended columns: `Date,High,Low,Close,Volume`.

Example:
```csv
Date,High,Low,Close,Volume
2026-01-05,151.2,148.8,150.4,120000
2026-01-06,153.1,150.0,152.5,145000
```
Column names are case-insensitive for the supported names. If Date is omitted, the app creates dates for display. Rows with invalid dates or prices are removed.

## DSA complexity
- Sliding Window monotonic deques: O(n) total time for all fixed-window min/max results.
- Heap top-K volatile days: O(n log k), where n is number of rows and k is requested top count.
- Segment Tree build: O(n); each range min/max query: O(log n).

## Project limitations / academic honesty
- The demo data is synthetic, not real market data.
- Signals are educational and do not predict future returns or constitute investment advice.
- In the college report, explain each algorithm and test it with at least one uploaded dataset. Cite any external dataset source you use.
