import heapq
import numpy as np
import pandas as pd



def make_demo_data(rows=120, seed=17):
    """Create a reproducible sample dataset so the project runs without an API key."""

    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(
        end=pd.Timestamp.today().normalize(),
        periods=rows
    )

    returns = rng.normal(loc=0.0006, scale=0.018, size=rows)

    close = 150 + np.cumsum(returns * 150)
    close = np.maximum(close, 5)

    daily_spread = rng.uniform(0.3, 2.5, size=rows)

    high = close + daily_spread
    low = np.maximum(0.01, close - daily_spread)

    volume = rng.integers(100_000, 2_000_000, size=rows)

    return pd.DataFrame({
        "Date": dates,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume
    })






def prepare_data(raw):
    """Validate columns, parse dates, clean prices, and calculate indicators."""
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("The CSV is empty.")
    df = raw.copy()
    # Handle common column naming variations.
    mapping = {}
    for col in df.columns:
        key = str(col).strip().lower().replace("_", " ").replace("-", " ")
        if key in ("date", "datetime", "timestamp"):
            mapping[col] = "Date"
        elif key in ("close", "closing price", "adj close", "adjclose", "price"):
            mapping[col] = "Close"
        elif key == "high":
            mapping[col] = "High"
        elif key == "low":
            mapping[col] = "Low"
        elif key in ("volume", "vol"):
            mapping[col] = "Volume"
    df = df.rename(columns=mapping)
    if "Close" not in df.columns:
        raise ValueError("A 'Close' column is required. Example: Date, Close.")
    if "Date" not in df.columns:
        df["Date"] = pd.date_range(end=pd.Timestamp.today().normalize(), periods=len(df))
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    for col in ("High", "Low", "Volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Date", "Close"])
    df = df[df["Close"] > 0].sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    if df.empty:
        raise ValueError("No valid rows remain after cleaning. Check Date and Close values.")
    if "High" not in df.columns:
        df["High"] = df["Close"]
    if "Low" not in df.columns:
        df["Low"] = df["Close"]
    if "Volume" not in df.columns:
        df["Volume"] = np.nan
    df["SMA_5"] = df["Close"].rolling(5, min_periods=1).mean()
    df["SMA_20"] = df["Close"].rolling(20, min_periods=1).mean()
    df["Daily_Change_%"] = df["Close"].pct_change().mul(100).fillna(0)
    return df.reset_index(drop=True)


def sliding_window_stats(df, window):
    """O(n) sliding-window stats using monotonic deques for min/max."""
    from collections import deque
    values = df["Close"].astype(float).tolist()
    min_q, max_q = deque(), deque()
    result = []
    for i, value in enumerate(values):
        while min_q and values[min_q[-1]] >= value:
            min_q.pop()
        min_q.append(i)
        while max_q and values[max_q[-1]] <= value:
            max_q.pop()
        max_q.append(i)
        left = i - window + 1
        while min_q and min_q[0] < left:
            min_q.popleft()
        while max_q and max_q[0] < left:
            max_q.popleft()
        if left >= 0:
            result.append({
                "Start Date": df["Date"].iloc[left].date().isoformat(),
                "End Date": df["Date"].iloc[i].date().isoformat(),
                "Window Min": round(values[min_q[0]], 2),
                "Window Max": round(values[max_q[0]], 2),
                "Window Range": round(values[max_q[0]] - values[min_q[0]], 2),
            })
    return pd.DataFrame(result)


def top_volatile_days(df, top_n=5):
    """Heap-based top-k ranking by absolute daily percentage movement."""
    heap = []
    for i in range(1, len(df)):
        move = float(df["Close"].iloc[i] - df["Close"].iloc[i - 1])
        pct = float(df["Daily_Change_%"].iloc[i])
        record = (abs(pct), i, move, pct)
        if len(heap) < top_n:
            heapq.heappush(heap, record)
        elif record[0] > heap[0][0]:
            heapq.heapreplace(heap, record)
    rows = []
    for abs_pct, i, move, pct in sorted(heap, reverse=True):
        rows.append({
            "Date": df["Date"].iloc[i].date().isoformat(),
            "Previous Close": round(float(df["Close"].iloc[i - 1]), 2),
            "Close": round(float(df["Close"].iloc[i]), 2),
            "Change": round(move, 2),
            "Change %": round(pct, 2),
            "Absolute Change %": round(abs_pct, 2),
        })
    return pd.DataFrame(rows)


class SegmentTree:
    """Segment tree supporting range minimum and maximum queries in O(log n)."""
    def __init__(self, values):
        if not values:
            raise ValueError("SegmentTree needs at least one value.")
        self.n = len(values)
        self.size = 1
        while self.size < self.n:
            self.size *= 2
        self.min_tree = [float("inf")] * (2 * self.size)
        self.max_tree = [float("-inf")] * (2 * self.size)
        for i, value in enumerate(values):
            self.min_tree[self.size + i] = float(value)
            self.max_tree[self.size + i] = float(value)
        for i in range(self.size - 1, 0, -1):
            self.min_tree[i] = min(self.min_tree[2*i], self.min_tree[2*i+1])
            self.max_tree[i] = max(self.max_tree[2*i], self.max_tree[2*i+1])

    def query(self, left, right):
        if left < 0 or right >= self.n or left > right:
            raise ValueError("Invalid query range.")
        left += self.size
        right += self.size
        mn, mx = float("inf"), float("-inf")
        while left <= right:
            if left % 2:
                mn = min(mn, self.min_tree[left])
                mx = max(mx, self.max_tree[left])
                left += 1
            if not right % 2:
                mn = min(mn, self.min_tree[right])
                mx = max(mx, self.max_tree[right])
                right -= 1
            left //= 2
            right //= 2
        return mn, mx


def detect_patterns(df):
    """Simple educational moving-average crossover signals."""
    if len(df) < 21:
        return ["Add at least 21 valid rows to evaluate 5-day and 20-day moving averages."]
    fast = df["Close"].rolling(5).mean()
    slow = df["Close"].rolling(20).mean()
    previous_diff = fast.iloc[-2] - slow.iloc[-2]
    current_diff = fast.iloc[-1] - slow.iloc[-1]
    signals = []
    if previous_diff <= 0 and current_diff > 0:
        signals.append("Possible bullish crossover: 5-day average crossed above 20-day average.")
    elif previous_diff >= 0 and current_diff < 0:
        signals.append("Possible bearish crossover: 5-day average crossed below 20-day average.")
    else:
        relation = "above" if current_diff > 0 else "below" if current_diff < 0 else "equal to"
        signals.append(f"No new crossover on the latest row. The 5-day average is {relation} the 20-day average.")
    recent = df["Close"].tail(20)
    if len(recent) >= 20:
        if df["Close"].iloc[-1] > recent.iloc[:-1].max():
            signals.append("Possible 20-row closing-price breakout: latest close is above the previous 19 closes.")
        elif df["Close"].iloc[-1] < recent.iloc[:-1].min():
            signals.append("Possible 20-row closing-price breakdown: latest close is below the previous 19 closes.")
    return signals
