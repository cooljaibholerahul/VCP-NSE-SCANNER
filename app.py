import io
import requests
import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(
    page_title="NSE VCP Scanner V2",
    page_icon="📈",
    layout="wide"
)

NSE_EQUITY_CSV = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"


@st.cache_data(ttl=86400, show_spinner=False)
def nse_symbols():
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,*/*",
        "Referer": "https://www.nseindia.com/"
    }

    r = requests.get(
        NSE_EQUITY_CSV,
        headers=headers,
        timeout=20
    )
    r.raise_for_status()

    df = pd.read_csv(io.BytesIO(r.content))

    col = "SYMBOL" if "SYMBOL" in df.columns else df.columns[0]

    return sorted(
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .unique()
        .tolist()
    )


@st.cache_data(ttl=900, show_spinner=False)
def get_data(symbol):

    df = yf.download(
        symbol + ".NS",
        period="2y",
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df[
        ["Open", "High", "Low", "Close", "Volume"]
    ].dropna()


def prepare(df):

    x = df.copy()

    x["SMA50"] = x.Close.rolling(50).mean()
    x["SMA150"] = x.Close.rolling(150).mean()
    x["SMA200"] = x.Close.rolling(200).mean()

    x["SMA200_20"] = x.SMA200.shift(20)

    tr = pd.concat(
        [
            x.High - x.Low,
            (x.High - x.Close.shift()).abs(),
            (x.Low - x.Close.shift()).abs()
        ],
        axis=1
    ).max(axis=1)

    x["ATRpct"] = (
        100 * tr.rolling(14).mean() / x.Close
    )

    x["Vol20"] = x.Volume.rolling(20).mean()

    x["High52"] = x.High.rolling(252).max()

    return x.dropna()


def contraction(series, n):

    w = series.iloc[-n:]

    return 100 * (w.max() - w.min()) / w.max()


def scan(symbol):

    df = get_data(symbol)

    if len(df) < 260:
        return None

    x = prepare(df)

    if len(x) < 260:
        return None

    r = x.iloc[-1]

    # RULE 1
    trend_template = bool(
        r.Close > r.SMA150 > r.SMA200
        and r.SMA200 > r.SMA200_20
        and r.SMA50 > r.SMA150
        and r.Close >= 0.75 * r.High52
    )

    # RULE 2
    c1 = contraction(x.Close, 40)
    c2 = contraction(x.Close, 25)
    c3 = contraction(x.Close, 15)

    tightening = bool(
        c2 < 0.85 * c1
        and c3 < 0.85 * c2
    )

    # RULE 3
    atr_now = x.ATRpct.iloc[-10:].mean()
    atr_old = x.ATRpct.iloc[-40:-10].mean()

    atr_contraction = bool(
        atr_now < atr_old
    )

    # RULE 4
    vol_now = x.Volume.iloc[-10:].mean()
    vol_old = x.Volume.iloc[-50:-10].mean()

    volume_dryup = bool(
        vol_now < 0.85 * vol_old
    )

    # RULE 5
    pivot = float(
        x.High.iloc[-20:].max()
    )

    pivot_distance = (
        100 * (pivot - r.Close) / pivot
    )

    near_pivot = bool(
        0 <= pivot_distance <= 8
    )

    # RULE 6
    breakout = bool(
        r.Close > pivot
        and r.Volume > 1.5 * r.Vol20
    )

    rules = {
        "Trend template": trend_template,
        "C1→C2→C3 tightening": tightening,
        "ATR contraction": atr_contraction,
        "Volume dry-up": volume_dryup,
        "Near pivot (≤8%)": near_pivot,
        "Breakout + 1.5x volume": breakout
    }

    passed = sum(rules.values())

    # ONLY SHOW STOCKS WITH 3 OR MORE PASSED RULES
    if passed < 3:
        return None

    return {
        "Symbol": symbol,

        "Passed": passed,

        "Total rules": 6,

        "Close": round(float(r.Close), 2),

        "C1 %": round(c1, 2),

        "C2 %": round(c2, 2),

        "C3 %": round(c3, 2),

        "ATR %": round(float(atr_now), 2),

        "Pivot": round(pivot, 2),

        "Pivot distance %":
            round(float(pivot_distance), 2),

        "Trend template":
            "PASS" if trend_template else "FAIL",

        "C1→C2→C3 tightening":
            "PASS" if tightening else "FAIL",

        "ATR contraction":
            "PASS" if atr_contraction else "FAIL",

        "Volume dry-up":
            "PASS" if volume_dryup else "FAIL",

        "Near pivot":
            "PASS" if near_pivot else "FAIL",

        "Breakout + volume":
            "PASS" if breakout else "FAIL"
    }


# -----------------------------
# APP
# -----------------------------

st.title("📈 NSE — Minervini VCP Scanner V2")

st.caption(
    "Shows stocks that pass 3 or more rules, "
    "with PASS/FAIL shown for every rule."
)


try:

    symbols = nse_symbols()

    st.success(
        f"NSE equity universe loaded: {len(symbols)} symbols"
    )

except Exception:

    st.error(
        "NSE symbol list could not be loaded right now."
    )

    symbols = []


# SIDEBAR

with st.sidebar:

    st.subheader("Scan settings")

    max_scan = st.number_input(
        "Stocks to scan",
        min_value=50,
        max_value=max(len(symbols), 50),
        value=len(symbols) if symbols else 500,
        step=50
    )

    st.info(
        "Only stocks passing 3 or more rules "
        "will appear in the results."
    )

    st.warning(
        "Full NSE scanning can take time because "
        "historical data is requested for many stocks."
    )


# SCAN BUTTON

if symbols:

    if st.button(
        "🔎 SCAN NSE FOR VCP V2",
        type="primary",
        use_container_width=True
    ):

        selected = symbols[:int(max_scan)]

        rows = []

        errors = 0

        progress = st.progress(0)

        status = st.empty()

        for i, symbol in enumerate(selected, 1):

            status.write(
                f"Scanning {symbol} — "
                f"{i}/{len(selected)}"
            )

            try:

                result = scan(symbol)

                if result:
                    rows.append(result)

            except Exception:

                errors += 1

            progress.progress(
                i / len(selected)
            )

        status.empty()

        progress.empty()


        # RESULTS

        if rows:

            output = pd.DataFrame(rows)

            output = output.sort_values(
                [
                    "Passed",
                    "Breakout + volume",
                    "Near pivot"
                ],
                ascending=[
                    False,
                    False,
                    False
                ]
            )

            st.success(
                f"{len(output)} stocks passed "
                f"3 or more rules."
            )

            st.dataframe(
                output,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download V2 CSV",

                output.to_csv(
                    index=False
                ).encode(),

                "nse_vcp_v2_candidates.csv",

                "text/csv"
            )

        else:

            st.info(
                "No stock passed 3 or more "
                "rules in the scanned universe."
            )

        if errors:

            st.warning(
                f"{errors} symbols could not be read "
                f"and were skipped."
            )


# RULE DESCRIPTION

st.divider()

st.markdown(
"""
### V2 Rules

**Rule 1 — Trend Template**

Price > 150 SMA > 200 SMA,  
200 SMA rising,  
50 SMA > 150 SMA,  
price ≥ 75% of 52-week high.

**Rule 2 — VCP Contraction**

C1 → C2 → C3 progressively tighter.

**Rule 3 — ATR Contraction**

Recent ATR% is lower than the preceding period.

**Rule 4 — Volume Dry-up**

Recent average volume is below 85% of the preceding period.

**Rule 5 — Near Pivot**

Price is within 0–8% below the 20-day pivot.

**Rule 6 — Breakout + Volume**

Price breaks above pivot with volume ≥ 1.5× 20-day average.

---

### RESULT CONDITION

A stock is displayed only when it passes:

**3 or more out of 6 rules.**

The result also shows exactly which rules were **PASS** and which were **FAIL**.
"""
)

st.caption(
    "Screening tool only; not financial advice. "
    "Market data may be delayed, incomplete, or rate-limited."
    )
