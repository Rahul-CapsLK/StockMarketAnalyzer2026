import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from analyzer import (
    make_demo_data, prepare_data, sliding_window_stats,
    top_volatile_days, SegmentTree, detect_patterns
)

st.set_page_config(page_title="Stock Market Pattern Analyzer", page_icon="📈", layout="wide")
st.title("📈 Stock Market Pattern Analyzer")
st.caption("College Major Project | Sliding Window • Heap • Segment Tree | Python")

with st.sidebar:
    st.header("Data settings")
    source = st.radio("Choose data source", ["Demo dataset", "Upload CSV"])
    uploaded = None
    if source == "Upload CSV":
        uploaded = st.file_uploader("Upload stock CSV", type=["csv"])
    window = st.slider("Sliding-window size (days)", min_value=3, max_value=30, value=7)
    top_n = st.slider("Top volatile days to show", min_value=3, max_value=15, value=5)

try:
    if source == "Upload CSV":
        if uploaded is None:
            st.info("Upload a CSV to continue. Required column: Close. Date is recommended; High and Low are optional.")
            st.stop()
        raw = pd.read_csv(uploaded)
    else:
        raw = make_demo_data()

    df = prepare_data(raw)
except Exception as exc:
    st.error(f"Could not read this dataset: {exc}")
    st.stop()

if len(df) < window:
    st.warning(f"Your dataset has {len(df)} rows. Choose a window size of {max(3, min(30, len(df)))} or upload more rows.")
    window = max(3, min(30, len(df)))

stats = sliding_window_stats(df, window)
patterns = detect_patterns(df)
vol_days = top_volatile_days(df, top_n)
tree = SegmentTree(df["Close"].tolist())

first, last = float(df["Close"].iloc[0]), float(df["Close"].iloc[-1])
change = last - first
change_pct = (change / first * 100) if first else 0
daily_change = df["Close"].pct_change().dropna()
volatility = float(daily_change.std() * 100) if len(daily_change) > 1 else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest close", f"{last:,.2f}")
c2.metric("Overall change", f"{change:+,.2f}", f"{change_pct:+.2f}%")
c3.metric("Highest close", f"{df['Close'].max():,.2f}")
c4.metric("Daily volatility (std.)", f"{volatility:.2f}%")

st.subheader("Price trend")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close price", mode="lines"))
fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_5"], name="5-day SMA", mode="lines"))
fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_20"], name="20-day SMA", mode="lines"))
fig.update_layout(height=440, xaxis_title="Date", yaxis_title="Price", hovermode="x unified", legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

# Candlestick Chart
st.subheader("🕯️ Candlestick Chart")

candle_fig = go.Figure()

candle_fig.add_trace(
    go.Candlestick(
        x=df["Date"],
        open=df["Close"].shift(1).fillna(df["Close"]),
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Stock Price"
    )
)

candle_fig.update_layout(
    height=500,
    xaxis_title="Date",
    yaxis_title="Price",
    xaxis_rangeslider_visible=False
)

st.plotly_chart(candle_fig, use_container_width=True)


# Volume Analysis Chart
st.subheader("📊 Trading Volume Analysis")

if "Volume" in df.columns and df["Volume"].notna().any():
    volume_fig = go.Figure()

    volume_fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=df["Volume"],
            name="Trading Volume"
        )
    )

    volume_fig.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="Volume",
        showlegend=False
    )

    st.plotly_chart(volume_fig, use_container_width=True)
else:
    st.warning("Volume data is not available in this dataset.")

# Chart 2: Daily Price Change
st.subheader("📊 Daily Price Change (%)")

fig2 = go.Figure()

fig2.add_trace(
    go.Bar(
        x=df["Date"],
        y=df["Daily_Change_%"],
        name="Daily Change"
    )
)

fig2.update_layout(
    height=400,
    xaxis_title="Date",
    yaxis_title="Daily Change (%)",
    hovermode="x unified"
)

st.plotly_chart(fig2, use_container_width=True)


# Chart 3: Closing Price Distribution
st.subheader("📈 Closing Price Distribution")

fig3 = go.Figure()

fig3.add_trace(
    go.Histogram(
        x=df["Close"],
        nbinsx=20,
        name="Closing Price"
    )
)

fig3.update_layout(
    height=400,
    xaxis_title="Closing Price",
    yaxis_title="Number of Days"
)

st.plotly_chart(fig3, use_container_width=True)


left, right = st.columns(2)
with left:
    st.subheader(f"Sliding Window: {window}-day range")
    st.write("The sliding-window algorithm calculates the minimum, maximum, and range for each consecutive window.")
    st.dataframe(stats.tail(15), use_container_width=True, hide_index=True)
with right:
    st.subheader("Pattern signals")
    if patterns:
        for item in patterns:
            st.info(item)
    else:
        st.write("Not enough data to identify a moving-average crossover. This is an analysis signal, not a prediction.")
    st.caption("Signals are educational indicators only and are not financial advice.")

st.subheader("Heap: Most volatile days")
st.write("A heap-based ranking selects days with the largest absolute daily price movement.")
st.dataframe(vol_days, use_container_width=True, hide_index=True)

st.subheader("Segment Tree: Range query")
st.write("Query the minimum and maximum closing price over any selected row interval. The end index is inclusive.")
max_idx = len(df) - 1
q1, q2 = st.columns(2)
with q1:
    start_idx = st.number_input("Start row index", min_value=0, max_value=max_idx, value=0, step=1)
with q2:
    end_idx = st.number_input("End row index", min_value=0, max_value=max_idx, value=max_idx, step=1)
if start_idx > end_idx:
    st.error("Start index must be less than or equal to end index.")
else:
    qmin, qmax = tree.query(int(start_idx), int(end_idx))
    st.success(f"Rows {int(start_idx)}–{int(end_idx)} | Minimum close: {qmin:,.2f} | Maximum close: {qmax:,.2f} | Range: {qmax-qmin:,.2f}")

st.subheader("Cleaned dataset")
st.dataframe(df[["Date", "Close", "SMA_5", "SMA_20", "Daily_Change_%"]], use_container_width=True, hide_index=True)
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download analyzed data as CSV", data=csv, file_name="analyzed_stock_data.csv", mime="text/csv")

with st.expander("How the DSA concepts are used"):
    st.markdown("""
    - **Sliding Window:** computes rolling minimum, maximum and range over a fixed number of days.
    - **Heap:** keeps the top days ranked by absolute daily price change.
    - **Segment Tree:** answers range minimum/maximum queries over closing prices.
    - **Pandas/Plotly:** data cleaning, moving averages and interactive charts.
    """)

# Market Trend Summary
st.subheader("📈 Market Trend Summary")

if len(df) >= 20:
    latest = df.iloc[-1]

    close_price = latest["Close"]
    sma_5 = latest["SMA_5"]
    sma_20 = latest["SMA_20"]

    st.metric("Latest Closing Price", f"{close_price:.2f}")
    st.metric("5-Day Moving Average", f"{sma_5:.2f}")
    st.metric("20-Day Moving Average", f"{sma_20:.2f}")

    if sma_5 > sma_20 and close_price > sma_5:
        st.success("🟢 Bullish Trend: Price is showing upward momentum.")

    elif sma_5 < sma_20 and close_price < sma_5:
        st.error("🔴 Bearish Trend: Price is showing downward momentum.")

    else:
        st.info("🟡 Sideways/Mixed Trend: No clear trend from these indicators.")

else:
    st.warning("At least 20 data rows are required to analyze the trend.")
