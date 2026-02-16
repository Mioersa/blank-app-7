import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

st.set_page_config(page_title="Options Data Analyzer", layout="wide")
st.title("📊 OPTIONS ANALYZER – Δ Metrics + Weighted Strength Signals")

# ============ Upload ============
files = st.file_uploader("Upload option‑chain CSVs", type=["csv"], accept_multiple_files=True)
if not files:
    st.info("👆 Upload CSVs first")
    st.stop()

def parse_time(name):
    m = re.search(r"_(\d{2})(\d{2})(\d{4})_(\d{2})(\d{2})(\d{2})", name)
    if not m: return None
    d, mo, y, h, mi, s = m.groups()
    return f"{d}-{mo}-{y} {h}:{mi}:{s}"

frames=[]
for f in files:
    d=pd.read_csv(f)
    ts=parse_time(f.name)
    if ts: d["timestamp"]=pd.to_datetime(ts)
    frames.append(d)
df=pd.concat(frames).sort_values("timestamp").reset_index(drop=True)

# ============ Compute Deltas ============
for pre in ["CE_","PE_"]:
    req = [f"{pre}totalTradedVolume", f"{pre}openInterest", f"{pre}lastPrice", f"{pre}impliedVolatility", f"{pre}strikePrice"]
    if not all(c in df.columns for c in req): continue
    df = df.groupby(f"{pre}strikePrice", group_keys=False).apply(
        lambda g: g.assign(
            **{
                f"{pre}volChange": g[f"{pre}totalTradedVolume"].diff(),
                f"{pre}oiChange": g[f"{pre}openInterest"].diff(),
                f"{pre}priceChange": g[f"{pre}lastPrice"].diff(),
                f"{pre}%Return": g[f"{pre}lastPrice"].pct_change()*100,
            }
        )
    )

# ============ Derived Metrics ============
if {"CE_openInterest","PE_openInterest"}.issubset(df.columns):
    df["OI_imbalance"]=(df["CE_openInterest"]-df["PE_openInterest"]) / (df["CE_openInterest"]+df["PE_openInterest"]+1e-9)

# ============ Rolling correlation (trend persistence) ============
window = st.sidebar.slider("Rolling window (points)",5,30,10)
for pre in ["CE_","PE_"]:
    if f"{pre}priceChange" in df.columns and f"{pre}oiChange" in df.columns:
        df[f"{pre}r_ΔP_OI"]=(
            df.groupby(f"{pre}strikePrice")[ [f"{pre}priceChange",f"{pre}oiChange"] ]
            .apply(lambda g: g[f"{pre}priceChange"].rolling(window).corr(g[f"{pre}oiChange"]) )
            .reset_index(level=0,drop=True)
        )

# ============ Quadrant classification ============
def quadrant(p,oi):
    if p>0 and oi>0: return "🟢 Long Buildup"
    if p<0 and oi<0: return "🔴 Long Unwind"
    if p>0 and oi<0: return "🟡 Short Cover"
    if p<0 and oi>0: return "🔵 Short Buildup"
    return "⚪ Flat"
for pre in ["CE_","PE_"]:
    if f"{pre}priceChange" in df.columns and f"{pre}oiChange" in df.columns:
        df[f"{pre}Quadrant"]=[quadrant(p,o) for p,o in zip(df[f"{pre}priceChange"],df[f"{pre}oiChange"])]

# ============ RSI / Z‑score ============
lookback=14
for pre in ["CE_","PE_"]:
    if f"{pre}priceChange" in df.columns:
        diff=df[f"{pre}priceChange"].fillna(0)
        up=(diff.where(diff>0,0)).rolling(lookback).mean()
        down=(-diff.where(diff<0,0)).rolling(lookback).mean()
        RS=up/(down+1e-9)
        df[f"{pre}RSI"]=100-(100/(1+RS))
        df[f"{pre}Zscore"]=(df[f"{pre}lastPrice"]-df[f"{pre}lastPrice"].rolling(lookback).mean())/df[f"{pre}lastPrice"].rolling(lookback).std()

# ============ Composite Strength ============
scores=[]
for s in sorted(pd.to_numeric(df.get("CE_strikePrice",df.get("PE_strikePrice",pd.Series())),errors="coerce").dropna().unique()):
    row={"Strike":s}
    for pre,side in [("CE_","CE"),("PE_","PE")]:
        if f"{pre}priceChange" not in df.columns: continue
        d=df[df[f"{pre}strikePrice"]==s]
        if d.empty: continue
        corr_price_oi=d[f"{pre}priceChange"].corr(d[f"{pre}oiChange"])
        corr_price_vol=d[f"{pre}priceChange"].corr(d.get(f"{pre}volChange"))
        avg_r=df[f"{pre}r_ΔP_OI"].mean()
        avg_rsi=(d[f"{pre}RSI"]).mean()
        bias=(avg_rsi-50)/50  # centered ±1
        oiimb=df["OI_imbalance"].mean()
        # weighted composition
        strength=0.4*(corr_price_oi or 0)+0.2*(corr_price_vol or 0)+0.2*(avg_r or 0)+0.2*bias
        row[f"{side}_Strength"]=strength
    scores.append(row)

if not scores:
    st.warning("No computed strikes")
    st.stop()

out=pd.DataFrame(scores)
out["Bias"]=out.apply(lambda r: "🟢 Bull" if r.get("CE_Strength",0)>r.get("PE_Strength",0) else "🔴 Bear",axis=1)
bias_colors={"🟢 Bull":"#ccffcc","🔴 Bear":"#ffcccc"}
st.subheader("💪 Composite Strength Score (Weighted)")
st.dataframe(out.style.applymap(lambda v:f"background-color:{bias_colors.get(v,'')}",subset=["Bias"]).format(precision=3))

fig=px.bar(out,x="Strike",y=["CE_Strength","PE_Strength"],barmode="group",title="Composite Strength (CE vs PE)")
st.plotly_chart(fig,use_container_width=True)

overall = "🟢 Overall Bull Bias" if out["CE_Strength"].mean()>out["PE_Strength"].mean() else "🔴 Overall Bear Bias"
st.markdown(f"### {overall}")

# ============ Quick One‑liner signals ============
st.markdown("---")
st.subheader("⚡ Live One‑Liner Signals (summary snapshot)")
latest=df.groupby("timestamp").tail(1)
if not latest.empty:
    ce=latest.iloc[-1]
    msg=[]
    for pre,label in [("CE_","🟢 Calls"),("PE_","🔴 Puts")]:
        if f"{pre}RSI" in latest:
            rsi=ce[f"{pre}RSI"]
            if rsi>60: sig="Bullish momentum"
            elif rsi<40: sig="Bearish momentum"
            else: sig="Neutral"
            msg.append(f"{label}: RSI {rsi:.1f} → {sig}")
    st.markdown("<br>".join(msg),unsafe_allow_html=True)
else:
    st.info("Signals will appear once data loaded.")
