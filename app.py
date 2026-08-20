import io
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="NSE VCP Scanner V3",
    page_icon="📈",
    layout="wide",
)

NSE_EQUITY_CSV = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
LOOKBACK = 100
DEFAULT_BATCH = 50
MIN_RULES = 3


@st.cache_data(ttl=86400, show_spinner=False)
def nse_symbols():
    r = requests.get(
        NSE_EQUITY_CSV,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/csv,*/*",
        },
        timeout=30,
    )
    r.raise_for_status()

    df = pd.read_csv(io.BytesIO(r.content))

    if "SYMBOL" not in df.columns:
        raise ValueError("NSE symbol file format has changed.")

    series_col = " SERIES" if " SERIES" in df.columns else "SERIES"
    if series_col in df.columns:
        df = df[df[series_col].astype(str).str.strip().eq("EQ")]

    return sorted(
        df["SYMBOL"]
        .astype(str)
        .str.strip()
        .dropna()
        .unique()
        .tolist()
    )


def clean_ohlcv(df):
    if df is None or len(df) < 120:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        if len(df.columns.get_level_values(1).unique()) == 1:
            df.columns = df.columns.get_level_values(0)

    wanted = ["Open", "High", "Low", "Close", "Volume"]
    lookup = {str(c).strip().title(): c for c in df.columns}

    if not all(x in lookup for x in wanted):
        return None

    out = df[[lookup[x] for x in wanted]].copy()
    out.columns = wanted
    out = out.dropna()

    if len(out) < 120:
        return None

    return out


def depth_percent(high, low):
    if high <= 0:
        return np.nan
    return (high - low) / high * 100.0


def find_contractions(df):
    """
    Detect distinct pullback/recovery structures inside the last 100
    daily sessions. A neighbourhood of 3 sessions is used around
    swing points so one sideways range is not split into many pieces.
    """
    d = df.tail(LOOKBACK).copy()

    if len(d) < LOOKBACK:
        return []

    highs = d["High"].to_numpy(float)
    lows = d["Low"].to_numpy(float)

    swing_highs = []
    swing_lows = []

    for i in range(3, len(d) - 3):
        if highs[i] >= highs[i - 3:i + 4].max():
            swing_highs.append(i)

        if lows[i] <= lows[i - 3:i + 4].min():
            swing_lows.append(i)

    candidates = []

    for hi in swing_highs:
        later_lows = [x for x in swing_lows if x > hi + 2]

        if not later_lows:
            continue

        lo = later_lows[0]
        depth = depth_percent(highs[hi], lows[lo])

        # Ignore insignificant noise and extreme breakdowns.
        if not np.isfinite(depth) or depth < 4 or depth > 45:
            continue

        later_highs = [x for x in swing_highs if x > lo + 2]

        if not later_highs:
            continue

        recovery_high = later_highs[0]

        # Require a meaningful recovery so the structure is not
        # simply an unfinished decline.
        recovery = (
            highs[recovery_high] - lows[lo]
        ) / max(highs[hi] - lows[lo], 1e-9)

        if recovery < 0.55:
            continue

        candidates.append((hi, lo, recovery_high, depth))

    candidates.sort(key=lambda x: x[2])

    # Keep non-overlapping structures.
    selected = []

    for c in candidates:
        if selected and c[0] <= selected[-1][2]:
            # If overlapping, keep the tighter structure.
            if c[3] < selected[-1][3]:
                selected[-1] = c
        else:
            selected.append(c)

    return selected[-5:]


def contraction_rule(contractions):
    """
    Final contraction rule:
    - 2 or 3 genuine contractions
    - later contraction depth must be smaller
    - structures must be separated, not pieces of one range
    """
    if len(contractions) < 2:
        return False, []

    recent = contractions[-3:]
    depths = [x[3] for x in recent]

    decreasing = all(
        depths[i] < depths[i - 1]
        for i in range(1, len(depths))
    )

    distinct = all(
        recent[i][0] > recent[i - 1][2]
        for i in range(1, len(recent))
    )

    return decreasing and distinct, depths


def volume_dryup(df, contractions):
    if len(contractions) < 2:
        return False

    recent = contractions[-3:]
    ratios = []

    volume = df["Volume"].tail(LOOKBACK)

    for hi, lo, _, _ in recent:
        segment = volume.iloc[hi:lo + 1]

        if len(segment) < 4:
            return False

        n = max(2, len(segment) // 3)

        early = segment.iloc[:n].median()
        late = segment.iloc[-n:].median()

        if early <= 0:
            return False

        ratios.append(late / early)

    return all(r <= 0.95 for r in ratios)


def price_near_high_rule(df, contractions):
    if not contractions:
        return False

    close = float(df["Close"].iloc[-1])
    recent = contractions[-3:]

    reference_highs = []

    for hi, _, recovery_high, _ in recent:
        reference_highs.append(float(df["High"].iloc[hi]))
        reference_highs.append(float(df["High"].iloc[recovery_high]))

    reference = max(reference_highs)

    distance = abs(close / reference - 1.0) * 100.0

    # The user's "0–10–20 from high" concept is treated as
    # "near the high", with 20% as the outer tolerance.
    return distance <= 20.0


def sma_rule(df):
    if len(df) < 110:
        return False

    close = float(df["Close"].iloc[-1])
    sma50 = float(df["Close"].rolling(50).mean().iloc[-1])
    sma100 = float(df["Close"].rolling(100).mean().iloc[-1])

    return bool(
        close > sma50
        and sma50 > sma100
    )


def ema_status(df):
    ema50 = df["Close"].ewm(span=50, adjust=False).mean().iloc[-1]
    ema100 = df["Close"].ewm(span=100, adjust=False).mean().iloc[-1]
    close = df["Close"].iloc[-1]

    return {
        "Price > 50 EMA": "YES" if close > ema50 else "NO",
        "Price > 100 EMA": "YES" if close > ema100 else "NO",
        "50 EMA > 100 EMA": "YES" if ema50 > ema100 else "NO",
    }


def cup_handle_indicator(df):
    """
    Cup-with-Handle is an indication only.
    It is NEVER used as a filtering rule.
    """
    if len(df) < 120:
        return False

    d = df.tail(180)

    if len(d) < 120:
        return False

    prices = d["Close"].to_numpy(float)
    n = len(prices)

    left = prices[: n // 3]
    middle = prices[n // 3: 2 * n // 3]
    right = prices[2 * n // 3:]

    left_peak = float(np.max(left))
    right_peak = float(np.max(right))
    bottom = float(np.min(middle))

    if left_peak <= 0 or right_peak <= 0:
        return False

    depth = (left_peak - bottom) / left_peak
    symmetry = abs(right_peak / left_peak - 1.0)

    # Broad structural indication, not a strict textbook classifier.
    if not (0.12 <= depth <= 0.55):
        return False

    if symmetry > 0.18:
        return False

    recent = prices[-25:]
    recent_peak = float(np.max(recent))
    current = float(prices[-1])

    handle_depth = (
        recent_peak - current
    ) / max(recent_peak, 1e-9)

    return bool(
        0.02 <= handle_depth <= 0.15
        and current >= bottom * 1.10
    )


def weekly_from_daily(df):
    weekly = df.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()

    return weekly


def analyse_stock(df):
    d = clean_ohlcv(df)

    if d is None:
        return None

    contractions = find_contractions(d)

    contraction_pass, depths = contraction_rule(contractions)
    volume_pass = volume_dryup(d, contractions)
    high_pass = price_near_high_rule(d, contractions)
    sma_pass = sma_rule(d)

    rules = {
        "R1: 2–3 genuine contractions": contraction_pass,
        "R2: Contraction volume dry-up": volume_pass,
        "R3: Price near contraction/previous high": high_pass,
        "R4: 50 SMA + 100 SMA trend": sma_pass,
    }

    passed = sum(rules.values())

    ema = ema_status(d)

    weekly = weekly_from_daily(d)

    return {
        "Close": round(float(d["Close"].iloc[-1]), 2),
        "Contractions": len(contractions),
        "Depths": " → ".join(
            f"{x:.1f}%" for x in depths
        ),
        "Rules Passed": passed,
        "R1": "PASS" if rules["R1: 2–3 genuine contractions"] else "FAIL",
        "R2": "PASS" if rules["R2: Contraction volume dry-up"] else "FAIL",
        "R3": "PASS" if rules["R3: Price near contraction/previous high"] else "FAIL",
        "R4": "PASS" if rules["R4: 50 SMA + 100 SMA trend"] else "FAIL",
        **ema,
        "Daily C&H": "YES" if cup_handle_indicator(d) else "NO",
        "Weekly C&H": "YES" if cup_handle_indicator(weekly) else "NO",
    }


def download_batch(symbols):
    tickers = [f"{s}.NS" for s in symbols]

    return yf.download(
        tickers=tickers,
        period="9mo",
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )


st.title("📈 NSE — Minervini VCP Scanner V3")

st.caption(
    "Rule-based investment research scanner. "
    "It does not predict prices or place trades."
)

try:
    symbols = nse_symbols()
    st.success(
        f"NSE equity universe loaded: {len(symbols)} symbols"
    )
except Exception as e:
    st.error(f"Could not load NSE symbols: {e}")
    st.stop()


with st.sidebar:
    st.subheader("Scanner settings")

    batch_size = st.number_input(
        "Batch size",
        min_value=10,
        max_value=100,
        value=DEFAULT_BATCH,
        step=10,
    )

    min_rules = st.number_input(
        "Minimum rules passed",
        min_value=3,
        max_value=4,
        value=MIN_RULES,
        step=1,
    )

    st.info(
        "All NSE symbols are scanned in batches. "
        "A batch is only a technical download group; "
        "the final result combines all batches."
    )


if st.button(
    "🔎 SCAN ALL NSE FOR VCP V3",
    type="primary",
    use_container_width=True,
):
    results = []

    total = len(symbols)
    progress = st.progress(0)
    status = st.empty()

    for start in range(0, total, int(batch_size)):
        batch = symbols[
            start:start + int(batch_size)
        ]

        status.write(
            f"Scanning {start + 1}–"
            f"{min(start + len(batch), total)} "
            f"of {total} stocks..."
        )

        try:
            raw = download_batch(batch)

            for symbol in batch:
                try:
                    ticker = f"{symbol}.NS"

                    if isinstance(raw.columns, pd.MultiIndex):
                        level0 = raw.columns.get_level_values(0)

                        if ticker not in level0:
                            continue

                        one = raw[ticker].copy()
                    else:
                        one = raw.copy()

                    analysis = analyse_stock(one)

                    if (
                        analysis is not None
                        and analysis["Rules Passed"] >= int(min_rules)
                    ):
                        results.append({
                            "Symbol": symbol,
                            **analysis,
                        })

                except Exception:
                    continue

        except Exception as e:
            st.warning(
                f"Batch starting at {start + 1} failed: {e}"
            )

        progress.progress(
            min(1.0, (start + len(batch)) / total)
        )

        time.sleep(0.2)

    status.write("Scan complete.")

    if results:
        result_df = pd.DataFrame(results)

        result_df = result_df.sort_values(
            by=["Rules Passed", "Symbol"],
            ascending=[False, True],
        )

        st.success(
            f"{len(result_df)} stocks passed at least "
            f"{int(min_rules)} rules."
        )

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download results CSV",
            result_df.to_csv(index=False).encode("utf-8"),
            "nse_vcp_v3_results.csv",
            "text/csv",
        )

    else:
        st.warning(
            "No stocks passed the minimum rule count. "
            "This does not mean the market has no VCPs; "
            "it means the current mechanical rules/data "
            "did not produce a qualifying result."
        )


st.divider()

st.subheader("Final V3 rules")

st.markdown("""
**R1 — Contractions**
- Daily chart.
- Last 100 trading sessions.
- 2 or 3 genuine contractions.
- Contractions must be separate swing structures.
- One sideways range must not be split into fake contractions.
- Later contraction depth should be tighter than the earlier one.

**R2 — Volume**
- Volume should progressively dry up inside the contraction structures.

**R3 — Price location**
- Current price should be near the contraction/previous high.
- The 0–10–20% proximity idea is implemented with 20% as the outer tolerance.

**R4 — Moving averages**
- Price > 50 SMA.
- 50 SMA > 100 SMA.

**Minimum result**
- Only stocks passing at least 3 of the 4 rules are displayed.
- Every displayed stock shows PASS/FAIL for every rule.

**EMA**
- EMA is informational only:
  - Price > 50 EMA
  - Price > 100 EMA
  - 50 EMA > 100 EMA

**Cup-with-Handle**
- Daily and weekly indications are shown after filtering.
- Cup-with-Handle is NOT used as a filter.
""")
