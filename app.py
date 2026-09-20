from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
UNIVERSE = BASE / "data" / "universe.csv"
DEMO = BASE / "data" / "demo_results.csv"

st.set_page_config(
    page_title="日本株・株式レーダー",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.block-container {padding-top:.8rem; padding-bottom:3rem; max-width:1180px;}
.radar-hero {padding:1.1rem 1.2rem;border-radius:18px;background:linear-gradient(135deg,#0b1930,#123f5a);color:white;margin-bottom:.8rem;}
.radar-hero h1 {margin:0;font-size:2rem}.radar-hero p{margin:.4rem 0 0;color:#d4ecf4}
[data-testid="stMetric"] {background:#f5f8fb;border:1px solid #dce5ec;padding:.55rem;border-radius:12px;}
[data-testid="stMetricValue"] {font-size:1.45rem;}
.small-note{font-size:.83rem;color:#667085}
@media (max-width:640px){
 .block-container{padding-left:.55rem;padding-right:.55rem;padding-top:.45rem}
 .radar-hero{padding:.85rem;border-radius:14px}.radar-hero h1{font-size:1.55rem}
 [data-testid="stMetricValue"]{font-size:1.2rem}
 h2{font-size:1.25rem!important} h3{font-size:1.1rem!important}
}
</style>
""",
    unsafe_allow_html=True,
)

LABELS = {
    "money": "資金流入",
    "revision": "上方修正・業績加速",
    "breakout": "新高値・ブレイクアウト",
    "growth": "連続増益・最高益",
    "dividend": "高配当・優良株",
}

DISPLAY = {
    "rank": "順位", "code": "コード", "company": "会社名", "price": "株価",
    "total_score": "総合点", "overlap": "上位部門数", "money_score": "資金流入",
    "revision_score": "上方修正", "breakout_score": "新高値", "growth_score": "連続増益",
    "dividend_score": "高配当", "volume_ratio": "出来高倍率", "return_5d": "5日騰落率%",
    "distance_52w_high": "52週高値乖離%", "dividend_yield": "配当利回り%",
    "market_cap_bil": "時価総額(10億円)", "operating_margin": "営業利益率%",
    "equity_ratio": "自己資本比率%", "forecast_revision": "会社予想修正率%",
    "sales_growth": "売上成長率%", "profit_growth": "利益成長率%", "roe": "ROE%",
    "data_status": "データ", "updated_at": "更新時刻",
}


def num(v, default=np.nan):
    try:
        if v is None or pd.isna(v):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def pct(v):
    v = num(v)
    if pd.isna(v):
        return np.nan
    return v * 100 if abs(v) <= 1.5 else v


def clip_score(v):
    return int(round(max(0, min(100, num(v, 0)))))


def score_row(row: pd.Series) -> dict:
    vr = num(row.get("volume_ratio"), 0)
    r5 = num(row.get("return_5d"), -99)
    r20 = num(row.get("return_20d"), -99)
    above20 = bool(row.get("above_ma20", False))
    money = 0
    money += 35 if vr >= 3 else 27 if vr >= 2 else 18 if vr >= 1.5 else 8 if vr >= 1.1 else 0
    money += 25 if 2 <= r5 <= 15 else 16 if r5 > 0 else 0
    money += 20 if r20 > 0 else 0
    money += 20 if above20 else 0

    rev = num(row.get("forecast_revision"))
    sg = num(row.get("sales_growth"))
    pg = num(row.get("profit_growth"))
    revision = 0
    revision += 45 if pd.notna(rev) and rev >= 10 else 32 if pd.notna(rev) and rev > 0 else 0
    revision += 25 if pd.notna(sg) and sg >= 10 else 15 if pd.notna(sg) and sg > 0 else 0
    revision += 30 if pd.notna(pg) and pg >= 15 else 20 if pd.notna(pg) and pg > 0 else 0

    high_gap = num(row.get("distance_52w_high"), -100)
    r60 = num(row.get("return_60d"), -99)
    breakout = 0
    breakout += 45 if high_gap >= -2 else 35 if high_gap >= -5 else 20 if high_gap >= -10 else 0
    breakout += 25 if vr >= 1.5 else 12 if vr >= 1.1 else 0
    breakout += 20 if r20 > 0 else 0
    breakout += 10 if r60 > 0 else 0

    consecutive = int(num(row.get("consecutive_growth_years"), 0))
    record = bool(row.get("record_profit", False))
    margin = num(row.get("operating_margin"))
    growth = 0
    growth += 45 if consecutive >= 5 else 35 if consecutive >= 3 else 20 if consecutive >= 2 else 0
    growth += 30 if record else 0
    growth += 15 if pd.notna(pg) and pg >= 10 else 8 if pd.notna(pg) and pg > 0 else 0
    growth += 10 if pd.notna(margin) and margin >= 10 else 5 if pd.notna(margin) and margin > 0 else 0

    dy = num(row.get("dividend_yield"))
    payout = num(row.get("payout_ratio"))
    equity = num(row.get("equity_ratio"))
    fcf = num(row.get("free_cash_flow_bil"))
    cap = num(row.get("market_cap_bil"))
    cuts = num(row.get("dividend_cuts_5y"))
    dividend = 0
    dividend += 30 if pd.notna(dy) and 4.5 <= dy <= 6.5 else 22 if pd.notna(dy) and 3.5 <= dy <= 7 else 0
    dividend += 20 if pd.notna(payout) and 25 <= payout <= 65 else 10 if pd.notna(payout) and 0 < payout <= 80 else 0
    dividend += 15 if pd.notna(equity) and equity >= 50 else 8 if pd.notna(equity) and equity >= 30 else 0
    dividend += 15 if pd.notna(fcf) and fcf > 0 else 0
    dividend += 10 if pd.notna(cap) and cap >= 500 else 5 if pd.notna(cap) and cap >= 100 else 0
    dividend += 10 if pd.notna(cuts) and cuts == 0 else 5 if pd.notna(cuts) and cuts <= 1 else 0

    scores = {
        "money_score": clip_score(money), "revision_score": clip_score(revision),
        "breakout_score": clip_score(breakout), "growth_score": clip_score(growth),
        "dividend_score": clip_score(dividend),
    }
    overlap = sum(v >= 70 for v in scores.values())
    top3 = sorted(scores.values(), reverse=True)[:3]
    total = round(0.55 * top3[0] + 0.30 * top3[1] + 0.15 * top3[2] + overlap * 3, 1)
    return {**scores, "overlap": overlap, "total_score": min(total, 100)}


def score_frame(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy()
    for c in ["code", "company", "ticker"]:
        if c not in d: d[c] = ""
        d[c] = d[c].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    calculated = d.apply(lambda r: pd.Series(score_row(r)), axis=1)
    for c in calculated.columns:
        d[c] = calculated[c]
    d = d.sort_values(["total_score", "overlap"], ascending=[False, False]).reset_index(drop=True)
    d.insert(0, "rank", range(1, len(d) + 1))
    return d


def safe_statement_value(frame, names, period_idx=0):
    if frame is None or frame.empty:
        return np.nan
    for name in names:
        if name in frame.index:
            values = pd.to_numeric(frame.loc[name], errors="coerce").dropna()
            if len(values) > period_idx:
                return float(values.iloc[period_idx])
    return np.nan


def growth_rate(current, previous):
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return np.nan
    return (current / abs(previous) - 1) * 100


def fetch_one(ticker: str, code: str, fallback_name: str) -> dict:
    import yfinance as yf

    obj = yf.Ticker(ticker)
    hist = obj.history(period="1y", auto_adjust=True)
    info = obj.info or {}
    if hist.empty:
        raise ValueError("price history unavailable")
    close = hist["Close"].dropna()
    volume = hist["Volume"].fillna(0)
    price = float(close.iloc[-1])
    avg20 = float(volume.tail(21).iloc[:-1].mean()) if len(volume) >= 21 else float(volume.mean())
    volume_ratio = float(volume.iloc[-1] / avg20) if avg20 > 0 else np.nan
    high52 = float(close.max())
    annual = obj.financials
    quarterly = obj.quarterly_financials
    rev0 = safe_statement_value(quarterly, ["Total Revenue"], 0)
    rev1 = safe_statement_value(quarterly, ["Total Revenue"], 4)
    op0 = safe_statement_value(quarterly, ["Operating Income", "EBIT"], 0)
    op1 = safe_statement_value(quarterly, ["Operating Income", "EBIT"], 4)

    revenues, profits = [], []
    if annual is not None and not annual.empty:
        for i in range(min(5, annual.shape[1])):
            revenues.append(safe_statement_value(annual, ["Total Revenue"], i))
            profits.append(safe_statement_value(annual, ["Operating Income", "EBIT"], i))
    consecutive = 0
    for i in range(min(len(revenues), len(profits)) - 1):
        if pd.notna(revenues[i]) and pd.notna(revenues[i+1]) and pd.notna(profits[i]) and pd.notna(profits[i+1]) and revenues[i] > revenues[i+1] and profits[i] > profits[i+1]:
            consecutive += 1
        else:
            break
    record_profit = bool(profits and pd.notna(profits[0]) and profits[0] >= np.nanmax(profits))
    divs = obj.dividends
    cuts = np.nan
    if divs is not None and len(divs):
        annual_div = divs.groupby(divs.index.year).sum().sort_index()
        annual_div = annual_div[annual_div.index < datetime.now().year].tail(6)
        cuts = int((annual_div.pct_change().dropna() < -.05).sum())

    def ret(days):
        return (price / float(close.iloc[-days]) - 1) * 100 if len(close) >= days else np.nan

    return {
        "code": code, "ticker": ticker,
        "company": info.get("longName") or info.get("shortName") or fallback_name,
        "price": price, "market_cap_bil": num(info.get("marketCap")) / 1e9,
        "volume_ratio": volume_ratio, "return_5d": ret(5), "return_20d": ret(20), "return_60d": ret(60),
        "above_ma20": price >= float(close.tail(20).mean()),
        "distance_52w_high": (price / high52 - 1) * 100,
        "forecast_revision": np.nan,
        "sales_growth": growth_rate(rev0, rev1), "profit_growth": growth_rate(op0, op1),
        "consecutive_growth_years": consecutive, "record_profit": record_profit,
        "dividend_yield": pct(info.get("dividendYield")), "payout_ratio": pct(info.get("payoutRatio")),
        "equity_ratio": np.nan, "free_cash_flow_bil": num(info.get("freeCashflow")) / 1e9,
        "dividend_cuts_5y": cuts, "operating_margin": pct(info.get("operatingMargins")),
        "roe": pct(info.get("returnOnEquity")), "data_status": "オンライン",
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_online(codes: tuple[str, ...]) -> pd.DataFrame:
    universe = pd.read_csv(UNIVERSE, dtype=str)
    universe = universe[universe["code"].isin(codes)]
    rows = []
    progress = st.progress(0, text="データ取得を開始します")
    for n, (_, r) in enumerate(universe.iterrows(), 1):
        progress.progress((n - 1) / max(len(universe), 1), text=f"{r['code']} {r['company']} を分析中")
        try:
            rows.append(fetch_one(r["ticker"], r["code"], r["company"]))
        except Exception as exc:
            rows.append({"code": r["code"], "ticker": r["ticker"], "company": r["company"], "data_status": f"取得失敗: {type(exc).__name__}"})
        time.sleep(.08)
    progress.progress(1.0, text="分析が完了しました")
    return pd.DataFrame(rows)


def show_table(d: pd.DataFrame, cols: list[str], height=520):
    available = [c for c in cols if c in d.columns]
    shown = d[available].rename(columns=DISPLAY)
    st.dataframe(shown, hide_index=True, width="stretch", height=height)


@st.cache_data
def load_demo():
    return pd.read_csv(DEMO, dtype={"code": str})


st.markdown('<div class="radar-hero"><h1>📡 日本株・株式レーダー</h1><p>5方向から探し、複数の検索で重なる有力候補を見つけます</p></div>', unsafe_allow_html=True)
st.warning("研究・検証用です。売買推奨ではありません。購入前に最新の決算短信・適時開示・会社IRを確認してください。")

with st.sidebar:
    st.header("データと表示")
    source = st.radio("データ", ["すぐ試す（見本データ）", "オンライン取得", "CSVを読み込む"])
    top_n = st.slider("表示件数", 10, 50, 20, 5)
    threshold = st.slider("各部門の上位判定", 50, 90, 70, 5)
    uploaded = st.file_uploader("分析済みCSV", type=["csv"])

if source == "CSVを読み込む" and uploaded is not None:
    raw = pd.read_csv(uploaded, dtype={"code": str})
elif source == "オンライン取得":
    universe = pd.read_csv(UNIVERSE, dtype=str)
    count = st.select_slider("今回取得する銘柄数", [10, 20, 30, 40], value=20)
    choices = st.multiselect("対象銘柄（未選択なら先頭から）", options=list(universe["code"] + " " + universe["company"]))
    selected_codes = tuple(x.split()[0] for x in choices) if choices else tuple(universe["code"].head(count))
    if st.button("最新データで検索", type="primary", width="stretch"):
        st.session_state["online_raw"] = fetch_online(selected_codes)
    raw = st.session_state.get("online_raw", load_demo())
    if "online_raw" not in st.session_state:
        st.info("「最新データで検索」を押すまでは見本データを表示しています。")
else:
    raw = load_demo()

df = score_frame(raw)
# The threshold is user-adjustable, so recompute overlap without altering category scores.
score_cols = [f"{x}_score" for x in LABELS]
df["overlap"] = (df[score_cols] >= threshold).sum(axis=1)
df = df.sort_values(["overlap", "total_score"], ascending=[False, False]).reset_index(drop=True)
df["rank"] = range(1, len(df) + 1)

top = df.iloc[0] if len(df) else pd.Series(dtype=object)
c1, c2, c3, c4 = st.columns(4)
c1.metric("分析銘柄", f"{len(df)}社")
c2.metric("複数部門で上位", f"{int((df['overlap'] >= 2).sum())}社")
c3.metric("総合1位", f"{top.get('code','—')} {top.get('company','—')}")
c4.metric("1位の総合点", f"{num(top.get('total_score'),0):.1f}")

tabs = st.tabs(["🏆 総合", "💹 資金流入", "📈 上方修正", "🚀 新高値", "🌱 連続増益", "💴 高配当", "🔎 銘柄詳細"])

with tabs[0]:
    st.subheader("複数の検索で重なる銘柄")
    st.caption(f"各部門{threshold}点以上を『上位』として数えます。総合点は得意な上位3部門を重視します。")
    show_table(df.head(top_n), ["rank","code","company","total_score","overlap",*score_cols])

section_map = [
    (1,"money",["volume_ratio","return_5d","return_20d"]),
    (2,"revision",["forecast_revision","sales_growth","profit_growth"]),
    (3,"breakout",["distance_52w_high","volume_ratio","return_20d"]),
    (4,"growth",["consecutive_growth_years","record_profit","profit_growth","operating_margin"]),
    (5,"dividend",["dividend_yield","payout_ratio","equity_ratio","free_cash_flow_bil"]),
]
for tab_index, key, extras in section_map:
    with tabs[tab_index]:
        st.subheader(LABELS[key] + "ランキング")
        part = df.sort_values(f"{key}_score", ascending=False).head(top_n).copy()
        part["rank"] = range(1, len(part)+1)
        show_table(part, ["rank","code","company",f"{key}_score",*extras,"data_status"])

with tabs[6]:
    st.subheader("銘柄ごとの5方向評価")
    options = [f"{r.code}  {r.company}" for r in df.itertuples()]
    selected = st.selectbox("証券コードまたは会社名", options)
    row = df.iloc[options.index(selected)]
    st.markdown(f"### {row['code']}　{row['company']}")
    cols = st.columns(5)
    for col, key in zip(cols, LABELS):
        col.metric(LABELS[key], f"{int(row[f'{key}_score'])}点")
    detail_cols = ["price","total_score","overlap","volume_ratio","return_5d","distance_52w_high","sales_growth","profit_growth","dividend_yield","operating_margin","roe","updated_at"]
    detail = pd.DataFrame({
        "項目": [DISPLAY.get(c, c) for c in detail_cols],
        "値": ["—" if pd.isna(row.get(c, np.nan)) else str(row.get(c)) for c in detail_cols],
    })
    st.dataframe(detail, hide_index=True, width="stretch")

st.divider()
csv = df.rename(columns=DISPLAY).to_csv(index=False).encode("utf-8-sig")
st.download_button("検索結果をCSVで保存", csv, f"stock_radar_{datetime.now():%Y%m%d}.csv", "text/csv", width="stretch")
st.caption("会社予想修正率は、適時開示等からCSVで入力した場合に反映されます。オンライン取得だけでは未取得（欠損）になる場合があります。")
