import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sqlalchemy import create_engine

# ---- 1. Load Stock Data From MySQL ----
@st.cache_data
def load_data():
    engine = create_engine("mysql+pymysql://root:root@localhost/stock_csv")
    with engine.connect() as conn:
        tables = pd.read_sql("SHOW TABLES", conn)
        dfs = []
        for table_name in tables.iloc[:, 0]:
            query = f"SELECT * FROM `{table_name}`"
            df_temp = pd.read_sql(query, conn, parse_dates=["date"])
            df_temp['ticker'] = table_name.lower()
            dfs.append(df_temp)
        df = pd.concat(dfs, ignore_index=True)
    return df

# ---- Manual Sector Mapping ----
nifty_sector_map = {
    "adaniports": "Services", "adanienterprises": "Services", "adani_green": "Power",
    "adani_total_gas": "Oil & Gas", "adani_power": "Power", "asianpaints": "Consumer Goods",
    "axisbank": "Banking", "baidulifesciences": "Pharma", "bharti_airtel": "Services",
    "bajaj_auto": "Auto", "bajaj_finance": "Finance", "bajaj_finserv": "Finance",
    "bharatforge": "Auto", "bharat_pe": "Services", "bharat_heavy_electricals": "Industrial",
    "bharat_petroleum": "Oil & Gas", "bhp": "Mining", "bpcl": "Oil & Gas", "castrol_india": "Oil & Gas",
    "cipla": "Pharma", "coal_india": "Mining", "divis_labs": "Pharma", "dr_reddys": "Pharma",
    "eicher_motors": "Auto", "gail": "Oil & Gas", "grasim_industries": "Industrial",
    "hcl_technologies": "Technology", "hdfc_bank": "Banking", "hdfc": "Finance",
    "hero_motocorp": "Auto", "hindalco_industries": "Metals", "hindustan_unilever": "Consumer Goods",
    "icici_bank": "Banking", "indusind_bank": "Banking", "infosys": "Technology",
    "jsw_steel": "Metals", "kotak_mahindra_bank": "Banking", "larsen_toubro": "Industrial",
    "mahindra": "Auto", "maruti_suzuki": "Auto", "ntpc": "Power", "ongc": "Oil & Gas",
    "power_grid": "Power", "reliance": "Oil & Gas", "sbicard": "Finance",
    "state_bank_of_india": "Banking", "sun_pharma": "Pharma",
    "tata_consultancy_services": "Technology", "tata_motors": "Auto", "tata_steel": "Metals",
    "tech_mahindra": "Technology", "titans": "Consumer Goods", "ultratech_cement": "Cement", "wipro": "Technology"
}

# ---- Preprocessing ----
df = load_data()
df["ticker"] = df["ticker"].str.strip().str.lower()
df["sector"] = df["ticker"].map(nifty_sector_map).fillna("Unknown")
df = df.sort_values("date")

# Fix 1: pct_change fill_method=None to avoid FutureWarning
df["daily_return"] = df.groupby("ticker")["close"].pct_change(fill_method=None)

df["cumulative_return"] = df.groupby("ticker")["daily_return"].transform(lambda x: (1 + x.fillna(0)).cumprod())
df["month"] = df["date"].dt.to_period("M").astype(str)
latest = df.groupby("ticker").last()
first = df.groupby("ticker").first()
df_yearly = (latest["close"] - first["close"]) / first["close"]
df["yearly_return"] = df["ticker"].map(df_yearly)

# ---- Streamlit UI ----
st.set_page_config(layout="wide", page_title="📊 Stock Dashboard")
st.title("📈 Stock Analytics Dashboard")

# ---- KPI Section ----
st.header("📊 Key Metrics Overview")
col1, col2, col3 = st.columns(3)
col1.metric("✅ Green Stocks", df_yearly[df_yearly > 0].count(), f"{(df_yearly > 0).mean()*100:.1f}% positive")
col2.metric("❌ Red Stocks", df_yearly[df_yearly <= 0].count(), f"{(df_yearly <= 0).mean()*100:.1f}% negative")
col3.metric("💰 Avg Price", f"{latest['close'].mean():.2f}", f"{df['close'].pct_change(fill_method=None).mean()*100:.2f}% daily")

# ---- Download CSV for Yearly Return ----
csv_yearly = df_yearly.reset_index().rename(columns={0: 'yearly_return'}).to_csv(index=False).encode()
st.download_button("📥 Download Yearly Returns CSV", csv_yearly, "yearly_returns.csv", "text/csv")

# ---- Gainers and Losers ----
st.subheader("📈 Top Gainers & 📉 Losers")
top10 = df_yearly.sort_values(ascending=False).head(10).to_frame(name="Yearly Return")
bottom10 = df_yearly.sort_values().head(10).to_frame(name="Yearly Return")
col1, col2 = st.columns(2)
with col1:
    st.write("🚀 Top 10 Gainers")
    # Fix 2: use Styler.map instead of applymap
    st.dataframe(top10.style.format("{:.2%}").map(lambda v: "color: green" if v > 0 else "color: red"))
with col2:
    st.write("📉 Top 10 Losers")
    st.dataframe(bottom10.style.format("{:.2%}").map(lambda v: "color: green" if v > 0 else "color: red"))

# ---- Volatility Chart ----
st.header("⚡ Most Volatile Stocks")
volatility = df.groupby("ticker")["daily_return"].std().sort_values(ascending=False).head(10)

# Seaborn style and palette for light theme
sns.set_theme(style="whitegrid", palette="crest")

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=volatility.index, y=volatility.values, ax=ax)
ax.set_title("Top 10 Most Volatile Stocks")
ax.set_ylabel("Volatility (Std Dev)")
plt.xticks(rotation=45)
st.pyplot(fig)

# ---- Cumulative Return Plot ----
st.header("📈 Cumulative Performance")
top5 = df.groupby("ticker")["cumulative_return"].last().sort_values(ascending=False).head(5).index
df_top5 = df[df["ticker"].isin(top5)]
fig = px.line(df_top5, x="date", y="cumulative_return", color="ticker", title="Top 5 Cumulative Return",
              template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# ---- Sector Performance ----
st.header("🏭 Sector-wise Performance")
sector_perf = df.groupby("sector")["yearly_return"].mean().sort_values(ascending=False).reset_index()
fig = px.bar(sector_perf, x="sector", y="yearly_return", color="yearly_return",
             title="Average Yearly Return by Sector", template="plotly_white",
             color_continuous_scale=px.colors.sequential.Viridis)
st.plotly_chart(fig, use_container_width=True)

# ---- Correlation Heatmap ----
st.header("📊 Correlation Heatmap")
pivot = df.pivot_table(index="date", columns="ticker", values="close", aggfunc="mean").ffill()
returns = pivot.pct_change().dropna()
corr = returns.corr()

fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(corr, cmap="crest", ax=ax)
ax.set_title("Correlation of Daily Returns")
st.pyplot(fig)

# ---- Monthly Returns ----
st.header("🗓️ Monthly Gainers & Losers")
months = sorted(df["month"].unique())
selected_month = st.selectbox("Select Month", months)
monthly = df[df["month"] == selected_month]
monthly_return = monthly.groupby("ticker")["close"].agg(["first", "last"])
monthly_return["monthly_return"] = (monthly_return["last"] - monthly_return["first"]) / monthly_return["first"]
monthly_return = monthly_return.sort_values("monthly_return", ascending=False)

col1, col2 = st.columns(2)
with col1:
    st.subheader("🚀 Top 5 Monthly Gainers")
    st.bar_chart(monthly_return["monthly_return"].head(5))
with col2:
    st.subheader("📉 Top 5 Monthly Losers")
    st.bar_chart(monthly_return["monthly_return"].tail(5))

# ---- Download Monthly CSV ----
csv_month = monthly_return.reset_index().to_csv(index=False).encode()
st.download_button("📥 Download Monthly Returns CSV", csv_month, "monthly_returns.csv", "text/csv")

































