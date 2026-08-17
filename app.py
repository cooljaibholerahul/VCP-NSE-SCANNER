import io
import requests
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="NSE VCP Scanner", page_icon="📈", layout="wide")

NSE_EQUITY_CSV = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"

@st.cache_data(ttl=86400, show_spinner=False)
def nse_symbols():
    headers = {"User-Agent":"Mozilla/5.0","Accept":"text/csv,*/*","Referer":"https://www.nseindia.com/"}
    r = requests.get(NSE_EQUITY_CSV, headers=headers, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    col = "SYMBOL" if "SYMBOL" in df.columns else df.columns[0]
    return sorted(df[col].dropna().astype(str).str.strip().str.upper().unique().tolist())

@st.cache_data(ttl=900, show_spinner=False)
def get_data(symbol):
    df = yf.download(symbol+".NS", period="2y", interval="1d",
                     auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    cols=["Open","High","Low","Close","Volume"]
    return df[[c for c in cols if c in df.columns]].dropna()

def prepare(df):
    x=df.copy()
    x["SMA50"]=x.Close.rolling(50).mean()
    x["SMA150"]=x.Close.rolling(150).mean()
    x["SMA200"]=x.Close.rolling(200).mean()
    x["SMA200_20"]=x.SMA200.shift(20)
    tr=pd.concat([x.High-x.Low,(x.High-x.Close.shift()).abs(),
                  (x.Low-x.Close.shift()).abs()],axis=1).max(axis=1)
    x["ATRpct"]=100*tr.rolling(14).mean()/x.Close
    x["Vol20"]=x.Volume.rolling(20).mean()
    x["High52"]=x.High.rolling(252).max()
    return x.dropna()

def contraction(s,n):
    w=s.iloc[-n:]
    return 100*(w.max()-w.min())/w.max()

def scan(sym, threshold):
    df=get_data(sym)
    if len(df)<260: return None
    x=prepare(df)
    if len(x)<260: return None
    r=x.iloc[-1]
    trend=(r.Close>r.SMA150>r.SMA200 and r.SMA200>r.SMA200_20
           and r.SMA50>r.SMA150 and r.Close>=.75*r.High52)
    if not trend: return None

    c1,c2,c3=[contraction(x.Close,n) for n in (40,25,15)]
    atr_now=x.ATRpct.iloc[-10:].mean()
    atr_old=x.ATRpct.iloc[-40:-10].mean()
    vol_now=x.Volume.iloc[-10:].mean()
    vol_old=x.Volume.iloc[-50:-10].mean()
    pivot=x.High.iloc[-20:].max()
    dist=100*(pivot-r.Close)/pivot
    tight=(c2<.85*c1 and c3<.85*c2)
    atr=atr_now<atr_old
    dry=vol_now<.85*vol_old
    near=0<=dist<=8
    breakout=r.Close>pivot and r.Volume>1.5*r.Vol20

    score=0; reasons=[]
    if tight: score+=3; reasons.append("tightening contractions")
    if atr: score+=1; reasons.append("ATR contracting")
    if dry: score+=2; reasons.append("volume dry-up")
    if near: score+=1; reasons.append("near pivot")
    if breakout: score+=2; reasons.append("breakout + volume")
    if score<threshold: return None

    return {"Symbol":sym,"Close":round(float(r.Close),2),"Score":score,
            "C1 %":round(c1,2),"C2 %":round(c2,2),"C3 %":round(c3,2),
            "Pivot":round(float(pivot),2),"Pivot distance %":round(float(dist),2),
            "Breakout":"YES" if breakout else "No","Reason":", ".join(reasons)}

st.title("📈 NSE — Minervini VCP Scanner")
st.caption("Rule-based screening prototype. It does not predict prices or place trades.")

try:
    symbols=nse_symbols()
    st.success(f"NSE equity universe loaded: {len(symbols)} symbols")
except Exception as e:
    st.error("NSE symbol list could not be loaded right now.")
    st.caption("You can retry later; the scanner requires current NSE symbols.")
    symbols=[]

with st.sidebar:
    threshold=st.slider("Minimum VCP score",5,9,5)
    max_scan=st.number_input("Maximum stocks to scan in this test",min_value=50,
                             max_value=max(len(symbols),50),value=min(len(symbols),500),step=50)
    st.warning("Full-NSE scanning can take time because each stock needs market-history data.")

if symbols and st.button("🔎 SCAN NSE FOR VCP",type="primary",use_container_width=True):
    selected=symbols[:int(max_scan)]
    rows=[]
    bar=st.progress(0)
    status=st.empty()
    for i,sym in enumerate(selected,1):
        status.write(f"Scanning {sym} — {i}/{len(selected)}")
        try:
            z=scan(sym,threshold)
            if z: rows.append(z)
        except Exception:
            pass
        bar.progress(i/len(selected))
    status.empty(); bar.empty()
    if rows:
        out=pd.DataFrame(rows).sort_values(["Breakout","Score","Pivot distance %"],
                                           ascending=[False,False,True])
        st.success(f"{len(out)} candidates found.")
        st.dataframe(out,use_container_width=True,hide_index=True)
        st.download_button("⬇️ Download CSV",out.to_csv(index=False).encode(),
                           "nse_vcp_candidates.csv","text/csv")
    else:
        st.info("No candidates matched the current rules.")

st.divider()
st.caption("Data: Yahoo Finance history for the NSE symbols. Validate data and signals independently before risking capital.")
