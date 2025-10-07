import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import plotly.express as px

# ---------------- Page & Style ----------------
st.set_page_config(page_title="Healthcare vs Non-Healthcare Dashboard", page_icon="📊",
                   layout="wide", initial_sidebar_state="collapsed")

# Define colors
NAVY  = "#0b1f44"      # bars / gauge
GOLD  = "#f0b429"      # net line
BLUE  = "#1f77b4"      # gross line
GREEN = "#10b981"      # alt net
SLATE = "#334155"
GRID  = "#e5e7eb"
RED   = "#dc2626"
EMERALD = "#059669"

st.markdown("""
<style>
.main {padding: 0rem 1rem;}
h1 {color:#0b1f44;font-weight:800;letter-spacing:.2px;border-bottom:3px solid #2ecc71;padding-bottom:10px;}
h2 {color:#0b1f44;font-weight:700;margin-top:1.2rem;margin-bottom:.6rem;}
.kpi {background:#fff;border:1px solid #e6e6e6;border-radius:14px;padding:14px;}
.k-num {font-size:36px;font-weight:800;color:#0b1f44;line-height:1.0;}
.k-cap {font-size:13px;color:#6b7280;margin-top:4px;}
.stTabs [data-baseweb="tab-list"] {gap: 8px;}
.stTabs [data-baseweb="tab"] {height: 50px;padding-left: 24px;padding-right: 24px;background-color: #f8f9fa;border-radius: 8px 8px 0 0;}
.stTabs [aria-selected="true"] {background-color: #0b1f44;color: white;}
.metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);border-radius:12px;padding:20px;color:white;}
.perf-table {border-radius:8px;overflow:hidden;box-shadow:0 2px 4px rgba(0,0,0,0.1);}
.stDataFrame {
    width: 100% !important;
}
.stDataFrame > div {
    width: 100% !important;
}
.stDataFrame table {
    width: 100% !important;
    table-layout: auto !important;
}
.stDataFrame th {
    white-space: nowrap !important;
    padding: 8px 12px !important;
    font-size: 12px !important;
    background-color: #f8f9fa !important;
}
.stDataFrame td {
    white-space: nowrap !important;
    padding: 6px 10px !important;
    font-size: 12px !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Healthcare & Non-Healthcare Dashboard")

# ---------------- Config ----------------
OTP_TARGET = 95

# EMEA Countries (comprehensive list including common variations)
EMEA_COUNTRIES = {
    # Europe
    'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 
    'LT', 'LU', 'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE', 'GB', 'UK', 'NO', 'CH', 'IS',
    'AL', 'AD', 'AM', 'AZ', 'BA', 'BY', 'GE', 'XK', 'LI', 'MD', 'MC', 'ME', 'MK', 'RU', 'SM', 'RS', 
    'TR', 'UA', 'VA',
    # Middle East
    'AE', 'BH', 'EG', 'IQ', 'IR', 'IL', 'JO', 'KW', 'LB', 'OM', 'PS', 'QA', 'SA', 'SY', 'YE',
    # Africa
    'DZ', 'AO', 'BJ', 'BW', 'BF', 'BI', 'CM', 'CV', 'CF', 'TD', 'KM', 'CG', 'CD', 'DJ', 'GQ', 'ER',
    'ET', 'GA', 'GM', 'GH', 'GN', 'GW', 'CI', 'KE', 'LS', 'LR', 'LY', 'MG', 'MW', 'ML', 'MR', 'MU',
    'MA', 'MZ', 'NA', 'NE', 'NG', 'RW', 'ST', 'SN', 'SC', 'SL', 'SO', 'ZA', 'SS', 'SD', 'SZ', 'TZ',
    'TG', 'TN', 'UG', 'ZM', 'ZW'
}

# Healthcare keywords for identification (expanded list)
HEALTHCARE_KEYWORDS = [
    'pharma', 'medical', 'health', 'bio', 'clinical', 'hospital', 'diagnostic',
    'therapeut', 'laborator', 'patholog', 'imaging', 'surgical', 'oncolog',
    'cardio', 'neuro', 'radiol', 'genetic', 'genomic', 'molecular', 'cell',
    'tissue', 'organ', 'transplant', 'vaccine', 'antibod', 'protein', 'peptide',
    'life science', 'lifescience', 'medic', 'therap', 'diagnost', 'clinic',
    'patient', 'treatment', 'disease', 'drug', 'dose', 'isotope', 'radio',
    'nuclear', 'pet', 'spect', 'immuno', 'assay', 'reagent', 'specimen',
    'sample', 'blood', 'plasma', 'serum', 'biobank', 'cryo', 'stem',
    # Specific company names
    'marken', 'fisher', 'cardinal', 'patheon', 'organox', 'qiagen', 'abbott',
    'tosoh', 'leica', 'sophia', 'cerus', 'sirtex', 'lantheus', 'avid',
    'petnet', 'innervate', 'ndri', 'university', 'institut', 'pentec',
    'sexton', 'atomics', 'curium', 'medtronic', 'catalent', 'delpharm',
    'veracyte', 'eckert', 'ziegler', 'shine', 'altasciences', 'smiths detection',
    'onkos', 'biolabs', 'biosystem', 'life molecular', 'cerveau', 'meilleur',
    'samsung bio', 'agilent', 'panasonic avionics'
]

# Non-healthcare keywords (explicit exclusions)
NON_HEALTHCARE_KEYWORDS = [
    'airline', 'airport', 'cargo', 'freight', 'logistic', 'transport',
    'express', 'disney', 'pictures', 'aviation', 'aircraft', 'aerospace',
    'volaris', 'easyjet', 'lufthansa', 'delta', 'american airlines',
    'british airways', 'nippon', 'aeromexico', 'spairliners', 'universal',
    'paramount', 'productions', 'courier', 'forwarding', 'tmr global',
    'aeroplex', 'nova traffic', 'ups', 'endeavor air', 
    'storm aviation', 'adventures', 'hartford', 'tokyo electron', 'slipstick',
    'sealion production', 'heathrow courier', 'macaronesia', 'exnet service',
    'mnx global logistics', 'logical freight', 'concesionaria', 'vuela compania'
]

CTRL_REGEX = re.compile(r"\b(agent|del\s*agt|delivery\s*agent|customs|warehouse|w/house)\b", re.I)

# ---------------- Helper Functions ----------------
def _excel_to_dt(s: pd.Series) -> pd.Series:
    """Robust datetime: parse; if many NaT, try Excel serials."""
    out = pd.to_datetime(s, errors="coerce")
    if out.isna().mean() > 0.5:
        num  = pd.to_numeric(s, errors="coerce")
        out2 = pd.to_datetime("1899-12-30") + pd.to_timedelta(num, unit="D")
        out  = out.where(~out.isna(), out2)
    return out

def _get_target_series(df: pd.DataFrame) -> pd.Series | None:
    if "UPD DEL" in df.columns and df["UPD DEL"].notna().any():
        return df["UPD DEL"]
    if "QDT" in df.columns:
        return df["QDT"]
    return None

def _kfmt(n: float) -> str:
    if pd.isna(n): return ""
    try: n = float(n)
    except: return ""
    return f"{n/1000:.1f}K" if n >= 1000 else f"{n:.0f}"

def _revenue_fmt(n: float) -> str:
    """Format revenue numbers with K or M suffix"""
    if pd.isna(n): return ""
    try: n = float(n)
    except: return ""
    if n >= 1000000:
        return f"${n/1000000:.2f}M"
    elif n >= 1000:
        return f"${n/1000:.1f}K"
    else:
        return f"${n:.0f}"

def is_healthcare(account_name, sheet_name=None):
    """Determine if an account is healthcare-related."""
    if not account_name:
        return False
    EXCLUDE_FROM_HEALTHCARE = {"avid", "lantheus", "life"}  # accounts you want out of Healthcare
    lower = str(account_name).strip().lower()
    if any(excluded in lower for excluded in EXCLUDE_FROM_HEALTHCARE):
        return False
      
    # Special rules by sheet
    if sheet_name == 'AMS':
        return True  # AMS is always healthcare
    if sheet_name == 'Aviation SVC':
        return False  # Aviation SVC is always non-healthcare
    
    lower = str(account_name).lower()
    
    # Check for explicit non-healthcare first (higher priority)
    for keyword in NON_HEALTHCARE_KEYWORDS:
        if keyword in lower:
            return False
    
    # Then check for healthcare
    for keyword in HEALTHCARE_KEYWORDS:
        if keyword in lower:
            return True
    
    # Default to False for uncertain cases
    return False

def make_semi_gauge(title: str, value: float) -> go.Figure:
    """Semi-donut gauge with centered %."""
    v = max(0.0, min(100.0, 0.0 if pd.isna(value) else float(value)))
    fig = go.Figure()
    fig.add_trace(go.Pie(
        values=[v, 100 - v, 100],
        hole=0.75, sort=False, direction="clockwise", rotation=180,
        textinfo="none", showlegend=False,
        marker=dict(colors=[NAVY, "#d1d5db", "rgba(0,0,0,0)"])
    ))
    fig.add_annotation(text=f"{v:.2f}%", x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=26, color=NAVY, family="Arial Black"))
    fig.add_annotation(text=title, x=0.5, y=1.18, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=14, color=SLATE))
    fig.update_layout(margin=dict(l=10, r=10, t=36, b=0), height=180)
    return fig

def analyze_monthly_changes(df: pd.DataFrame, metric: str = 'TOTAL CHARGES'):
    """Analyze month-over-month changes for accounts"""
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    
    # Group by account and month
    monthly = df.groupby(['ACCT NM', 'Month_Display', 'Month_Sort']).agg({
        metric: 'sum',
        'PIECES': 'sum' if 'PIECES' in df.columns else 'size',
        '_pod': 'count'  # Volume count
    }).reset_index()
    monthly.columns = ['Account', 'Month', 'Month_Sort', 'Revenue', 'Pieces', 'Volume']
    
    # Sort by account and month
    monthly = monthly.sort_values(['Account', 'Month_Sort'])
    
    # Calculate month-over-month changes
    monthly['Revenue_Prev'] = monthly.groupby('Account')['Revenue'].shift(1)
    monthly['Volume_Prev'] = monthly.groupby('Account')['Volume'].shift(1)
    monthly['Pieces_Prev'] = monthly.groupby('Account')['Pieces'].shift(1)
    
    monthly['Revenue_Change'] = monthly['Revenue'] - monthly['Revenue_Prev']
    monthly['Revenue_Change_Pct'] = ((monthly['Revenue'] / monthly['Revenue_Prev']) - 1) * 100
    monthly['Volume_Change'] = monthly['Volume'] - monthly['Volume_Prev']
    monthly['Volume_Change_Pct'] = ((monthly['Volume'] / monthly['Volume_Prev']) - 1) * 100
    monthly['Pieces_Change'] = monthly['Pieces'] - monthly['Pieces_Prev']
    monthly['Pieces_Change_Pct'] = ((monthly['Pieces'] / monthly['Pieces_Prev']) - 1) * 100
    
    return monthly

def create_performance_tables(monthly_changes: pd.DataFrame, month: str, sector: str):
    """Create top/worst performance tables for a specific month with enhanced columns"""
    if monthly_changes.empty:
        return
    
    month_data = monthly_changes[monthly_changes['Month'] == month].copy()
    if month_data.empty:
        st.info(f"No data available for {month}")
        return
    
    month_data = month_data.dropna(subset=['Revenue_Change'])
    
    if month_data.empty:
        st.info(f"No month-over-month data available for {month}")
        return
    
    st.subheader(f"📊 {sector} Performance Analysis - {month}")
    
    # Revenue Performance with Volume columns
    st.markdown("### 💰 Revenue Performance (with Volume Changes)")
    
    # Top 10 Revenue Increases
    top_revenue = month_data.nlargest(10, 'Revenue_Change')[
        ['Account', 'Revenue', 'Revenue_Prev', 'Revenue_Change', 'Revenue_Change_Pct',
         'Volume', 'Volume_Prev', 'Volume_Change', 'Volume_Change_Pct']
    ].copy()
    
    if not top_revenue.empty:
        st.markdown("**🔝 Top 10 Revenue Increases**")
        top_revenue['Current Rev'] = top_revenue['Revenue'].apply(_revenue_fmt)
        top_revenue['Rev Change'] = top_revenue.apply(
            lambda x: f"+{_revenue_fmt(x['Revenue_Change'])} ({x['Revenue_Change_Pct']:.2f}%)" 
            if pd.notna(x['Revenue_Change_Pct']) else f"+{_revenue_fmt(x['Revenue_Change'])}", axis=1
        )
        top_revenue['Current Vol'] = top_revenue['Volume'].apply(lambda x: f"{x:,.0f}")
        top_revenue['Vol Change'] = top_revenue.apply(
            lambda x: f"+{x['Volume_Change']:,.0f} ({x['Volume_Change_Pct']:.2f}%)" 
            if pd.notna(x['Volume_Change_Pct']) and pd.notna(x['Volume_Change']) 
            else "N/A", axis=1
        )
        
        display_df = top_revenue[['Account', 'Current Rev', 'Rev Change', 'Current Vol', 'Vol Change']].head(10)
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=400
        )
    
    # Worst 10 Revenue Decreases
    worst_revenue = month_data.nsmallest(10, 'Revenue_Change')[
        ['Account', 'Revenue', 'Revenue_Prev', 'Revenue_Change', 'Revenue_Change_Pct',
         'Volume', 'Volume_Prev', 'Volume_Change', 'Volume_Change_Pct']
    ].copy()
    
    if not worst_revenue.empty:
        st.markdown("**📉 Top 10 Revenue Decreases**")
        worst_revenue['Current Rev'] = worst_revenue['Revenue'].apply(_revenue_fmt)
        worst_revenue['Rev Change'] = worst_revenue.apply(
            lambda x: f"{_revenue_fmt(x['Revenue_Change'])} ({x['Revenue_Change_Pct']:.2f}%)" 
            if pd.notna(x['Revenue_Change_Pct']) else f"{_revenue_fmt(x['Revenue_Change'])}", axis=1
        )
        worst_revenue['Current Vol'] = worst_revenue['Volume'].apply(lambda x: f"{x:,.0f}")
        worst_revenue['Vol Change'] = worst_revenue.apply(
            lambda x: f"{x['Volume_Change']:,.0f} ({x['Volume_Change_Pct']:.2f}%)" 
            if pd.notna(x['Volume_Change_Pct']) and pd.notna(x['Volume_Change']) 
            else "N/A", axis=1
        )
        
        display_df = worst_revenue[['Account', 'Current Rev', 'Rev Change', 'Current Vol', 'Vol Change']].head(10)
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=400
        )
    
    st.markdown("---")

    # ---------------- Chart: Net OTP by Pieces (POD) ----------------
    st.subheader(f"{tab_name}: Controllable (Net) OTP by Pieces — POD Month")
    if not pieces_pod.empty and not otp_pod.empty:
        mp = pieces_pod.merge(otp_pod[["Month_YYYY_MM","Net_OTP"]],
                              on="Month_YYYY_MM", how="left").sort_values("Month_Sort")
        x = mp["Month_Display"].tolist()
        y_pcs = mp["Pieces"].astype(float).tolist()
        y_net = mp["Net_OTP"].astype(float).tolist()

        figp = go.Figure()
        # Bar chart with values at bottom
        figp.add_trace(go.Bar(
            x=x, y=y_pcs, name="Pieces", 
            marker_color=NAVY,
            text=[_kfmt(v) for v in y_pcs],
            textposition="inside",
            textfont=dict(size=14, color="white", family="Arial Black"),
            textangle=0,
            yaxis="y",
            insidetextanchor="start"  # This anchors text at bottom of bar
        ))
        
        # OTP line
        figp.add_trace(go.Scatter(
            x=x, y=y_net, name="Net OTP",
            mode="lines+markers",
            line=dict(color=GOLD, width=3),
            marker=dict(size=10, color=GOLD),
            yaxis="y2"
        ))
        
        # Add OTP percentage labels above the line points
        for i, (xi, yi) in enumerate(zip(x, y_net)):
            if pd.notna(yi):
                figp.add_annotation(
                    x=xi, y=yi, xref="x", yref="y2",
                    text=f"<b>{yi:.2f}%</b>",
                    showarrow=False,
                    yshift=20,
                    font=dict(size=13, color="#111827", family="Arial Black"),
                    bgcolor="white",
                    bordercolor=GOLD,
                    borderwidth=1,
                    borderpad=4
                )
        
        # Target line
        figp.add_shape(
            type="line", x0=-0.5, x1=len(x)-0.5,
            y0=float(otp_target), y1=float(otp_target),
            xref="x", yref="y2",
            line=dict(color="red", dash="dash", width=2)
        )
        
        # Add target label
        figp.add_annotation(
            x=len(x)-0.5, y=float(otp_target),
            xref="x", yref="y2",
            text=f"Target: {otp_target}%",
            showarrow=False,
            xshift=-50,
            font=dict(size=12, color="red"),
            bgcolor="white"
        )
        
        figp.update_layout(
            height=520,
            hovermode="x unified",
            plot_bgcolor="white",
            margin=dict(l=40, r=40, t=40, b=80),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
            xaxis=dict(title="", tickangle=-30, tickmode="array", tickvals=x, ticktext=x, automargin=True),
            yaxis=dict(title="Pieces", side="left", gridcolor=GRID, showgrid=True),
            yaxis2=dict(title="Net OTP (%)", overlaying="y", side="right", range=[0, 120], showgrid=False),
            barmode="overlay"
        )
        st.plotly_chart(figp, use_container_width=True)
    else:
        st.info("No monthly PIECES available.")

    st.markdown("---")

    # ---------------- Chart: Gross vs Net OTP (POD) ----------------
    st.subheader(f"{tab_name}: Monthly OTP Trend (Gross vs Net) — POD Month")
    if not otp_pod.empty:
        otp_sorted = otp_pod.sort_values("Month_Sort")
        x       = otp_sorted["Month_Display"].tolist()
        gross_y = otp_sorted["Gross_OTP"].astype(float).tolist()
        net_y   = otp_sorted["Net_OTP"].astype(float).tolist()

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x, y=gross_y, mode="lines+markers", name="Gross OTP",
                                  line=dict(color=BLUE, width=3), marker=dict(size=8)))
        fig2.add_trace(go.Scatter(x=x, y=net_y, mode="lines+markers", name="Net OTP",
                                  line=dict(color=GREEN, width=3), marker=dict(size=8)))
        
        # Add percentage labels for Gross OTP
        for xi, yi in zip(x, gross_y):
            if pd.notna(yi):
                fig2.add_annotation(
                    x=xi, y=yi, xref="x", yref="y",
                    text=f"<b>{yi:.2f}%</b>",
                    showarrow=False,
                    yshift=20,
                    font=dict(size=12, color=BLUE),
                    bgcolor="rgba(255,255,255,0.8)"
                )
        
        # Add percentage labels for Net OTP
        for xi, yi in zip(x, net_y):
            if pd.notna(yi):
                fig2.add_annotation(
                    x=xi, y=yi, xref="x", yref="y",
                    text=f"<b>{yi:.2f}%</b>",
                    showarrow=False,
                    yshift=-20,
                    font=dict(size=12, color=GREEN),
                    bgcolor="rgba(255,255,255,0.8)"
                )
        
        # Target line
        fig2.add_shape(
            type="line", x0=-0.5, x1=len(x)-0.5,
            y0=float(otp_target), y1=float(otp_target),
            xref="x", yref="y",
            line=dict(color="red", dash="dash", width=2)
        )
        
        # Add target label
        fig2.add_annotation(
            x=len(x)-0.5, y=float(otp_target),
            xref

    # ---------------- Chart: Net OTP by Volume (POD) ----------------
    st.subheader(f"{tab_name}: Controllable (Net) OTP by Volume — POD Month")
    if not vol_pod.empty and not otp_pod.empty:
        mv = vol_pod.merge(otp_pod[["Month_YYYY_MM","Net_OTP"]],
                           on="Month_YYYY_MM", how="left").sort_values("Month_Sort")
        x = mv["Month_Display"].tolist()
        y_vol = mv["Volume"].astype(float).tolist()
        y_net = mv["Net_OTP"].astype(float).tolist()

        fig = go.Figure()
        # Bar chart with values at bottom
        fig.add_trace(go.Bar(
            x=x, y=y_vol, name="Volume (Rows)", 
            marker_color=NAVY,
            text=[_kfmt(v) for v in y_vol],
            textposition="inside",
            textfont=dict(size=14, color="white", family="Arial Black"),
            textangle=0,
            yaxis="y",
            insidetextanchor="start"  # This anchors text at bottom of bar
        ))
        
        # OTP line
        fig.add_trace(go.Scatter(
            x=x, y=y_net, name="Net OTP",
            mode="lines+markers", 
            line=dict(color=GOLD, width=3),
            marker=dict(size=10, color=GOLD),
            yaxis="y2"
        ))
        
        # Add OTP percentage labels above the line points
        for i, (xi, yi) in enumerate(zip(x, y_net)):
            if pd.notna(yi):
                fig.add_annotation(
                    x=xi, y=yi, xref="x", yref="y2",
                    text=f"<b>{yi:.2f}%</b>",
                    showarrow=False,
                    yshift=20,
                    font=dict(size=13, color="#111827", family="Arial Black"),
                    bgcolor="white",
                    bordercolor=GOLD,
                    borderwidth=1,
                    borderpad=4
                )
        
        # Target line
        fig.add_shape(
            type="line", x0=-0.5, x1=len(x)-0.5,
            y0=float(otp_target), y1=float(otp_target),
            xref="x", yref="y2", 
            line=dict(color="red", dash="dash", width=2)
        )
        
        # Add target label
        fig.add_annotation(
            x=len(x)-0.5, y=float(otp_target),
            xref="x", yref="y2",
            text=f"Target: {otp_target}%",
            showarrow=False,
            xshift=-50,
            font=dict(size=12, color="red"),
            bgcolor="white"
        )
        
        fig.update_layout(
            height=520, 
            hovermode="x unified", 
            plot_bgcolor="white",
            margin=dict(l=40, r=40, t=40, b=80),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
            xaxis=dict(title="", tickangle=-30, tickmode="array", tickvals=x, ticktext=x, automargin=True),
            yaxis=dict(title="Volume (Rows)", side="left", gridcolor=GRID, showgrid=True),
            yaxis2=dict(title="Net OTP (%)", overlaying="y", side="right", range=[0, 120], showgrid=False),
            barmode="overlay"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No monthly volume available.")

    st.markdown("---")
    
    # Volume Performance with Revenue columns
    st.markdown("### 📦 Volume Performance (with Revenue Changes)")
    
    volume_data = month_data.dropna(subset=['Volume_Change'])
    if not volume_data.empty:
        # Top 10 Volume Increases
        top_volume = volume_data.nlargest(10, 'Volume_Change')[
            ['Account', 'Volume', 'Volume_Change', 'Volume_Change_Pct',
             'Revenue', 'Revenue_Prev', 'Revenue_Change', 'Revenue_Change_Pct']
        ].copy()
        
        st.markdown("**🔝 Top 10 Volume Increases**")
        top_volume['Current Vol'] = top_volume['Volume'].apply(lambda x: f"{x:,.0f}")
        top_volume['Vol Change'] = top_volume.apply(
            lambda x: f"+{x['Volume_Change']:,.0f} ({x['Volume_Change_Pct']:.2f}%)" 
            if pd.notna(x['Volume_Change_Pct']) else f"+{int(x['Volume_Change'])}", axis=1
        )
        top_volume['Current Rev'] = top_volume['Revenue'].apply(_revenue_fmt)
        top_volume['Rev Change'] = top_volume.apply(
            lambda x: f"{_revenue_fmt(x['Revenue_Change'])} ({x['Revenue_Change_Pct']:.2f}%)" 
            if pd.notna(x['Revenue_Change_Pct']) and pd.notna(x['Revenue_Change']) 
            else "N/A", axis=1
        )
        
        display_df = top_volume[['Account', 'Current Vol', 'Vol Change', 'Current Rev', 'Rev Change']].head(10)
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        # Worst 10 Volume Decreases
        worst_volume = volume_data.nsmallest(10, 'Volume_Change')[
            ['Account', 'Volume', 'Volume_Change', 'Volume_Change_Pct',
             'Revenue', 'Revenue_Prev', 'Revenue_Change', 'Revenue_Change_Pct']
        ].copy()
        
        st.markdown("**📉 Top 10 Volume Decreases**")
        worst_volume['Current Vol'] = worst_volume['Volume'].apply(lambda x: f"{x:,.0f}")
        worst_volume['Vol Change'] = worst_volume.apply(
            lambda x: f"{x['Volume_Change']:,.0f} ({x['Volume_Change_Pct']:.2f}%)" 
            if pd.notna(x['Volume_Change_Pct']) else f"{int(x['Volume_Change'])}", axis=1
        )
        worst_volume['Current Rev'] = worst_volume['Revenue'].apply(_revenue_fmt)
        worst_volume['Rev Change'] = worst_volume.apply(
            lambda x: f"{_revenue_fmt(x['Revenue_Change'])} ({x['Revenue_Change_Pct']:.2f}%)" 
            if pd.notna(x['Revenue_Change_Pct']) and pd.notna(x['Revenue_Change']) 
            else "N/A", axis=1
        )
        
        display_df = worst_volume[['Account', 'Current Vol', 'Vol Change', 'Current Rev', 'Rev Change']].head(10)
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=400
        )
    
    st.markdown("---")
    
    # Pieces Performance with Revenue columns
    st.markdown("### 📋 Pieces Performance (with Revenue Changes)")
    
    pieces_data = month_data.dropna(subset=['Pieces_Change'])
    if not pieces_data.empty:
        # Top 10 Pieces Increases
        top_pieces = pieces_data.nlargest(10, 'Pieces_Change')[
            ['Account', 'Pieces', 'Pieces_Change', 'Pieces_Change_Pct',
             'Revenue', 'Revenue_Prev', 'Revenue_Change', 'Revenue_Change_Pct']
        ].copy()
        
        st.markdown("**🔝 Top 10 Pieces Increases**")
        top_pieces['Current Pieces'] = top_pieces['Pieces'].apply(lambda x: f"{x:,.0f}")
        top_pieces['Pieces Change'] = top_pieces.apply(
            lambda x: f"+{x['Pieces_Change']:,.0f} ({x['Pieces_Change_Pct']:.2f}%)" 
            if pd.notna(x['Pieces_Change_Pct']) else f"+{int(x['Pieces_Change'])}", axis=1
        )
        top_pieces['Current Rev'] = top_pieces['Revenue'].apply(_revenue_fmt)
        top_pieces['Rev Change'] = top_pieces.apply(
            lambda x: f"{_revenue_fmt(x['Revenue_Change'])} ({x['Revenue_Change_Pct']:.2f}%)" 
            if pd.notna(x['Revenue_Change_Pct']) and pd.notna(x['Revenue_Change']) 
            else "N/A", axis=1
        )
        
        display_df = top_pieces[['Account', 'Current Pieces', 'Pieces Change', 'Current Rev', 'Rev Change']].head(10)
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        # Worst 10 Pieces Decreases
        worst_pieces = pieces_data.nsmallest(10, 'Pieces_Change')[
            ['Account', 'Pieces', 'Pieces_Change', 'Pieces_Change_Pct',
             'Revenue', 'Revenue_Prev', 'Revenue_Change', 'Revenue_Change_Pct']
        ].copy()
        
        st.markdown("**📉 Top 10 Pieces Decreases**")
        worst_pieces['Current Pieces'] = worst_pieces['Pieces'].apply(lambda x: f"{x:,.0f}")
        worst_pieces['Pieces Change'] = worst_pieces.apply(
            lambda x: f"{x['Pieces_Change']:,.0f} ({x['Pieces_Change_Pct']:.2f}%)" 
            if pd.notna(x['Pieces_Change_Pct']) else f"{int(x['Pieces_Change'])}", axis=1
        )
        worst_pieces['Current Rev'] = worst_pieces['Revenue'].apply(_revenue_fmt)
        worst_pieces['Rev Change'] = worst_pieces.apply(
            lambda x: f"{_revenue_fmt(x['Revenue_Change'])} ({x['Revenue_Change_Pct']:.2f}%)" 
            if pd.notna(x['Revenue_Change_Pct']) and pd.notna(x['Revenue_Change']) 
            else "N/A", axis=1
        )
        
        display_df = worst_pieces[['Account', 'Current Pieces', 'Pieces Change', 'Current Rev', 'Rev Change']].head(10)
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=400
        )

# ---------------- IO Functions ----------------
@st.cache_data(show_spinner=False)
def read_and_combine_sheets(uploaded):
    """Read Excel and split into healthcare and non-healthcare dataframes."""
    try:
        # Read ALL sheets explicitly
        sheet_names = ['Aviation SVC', 'MNX Charter', 'AMS', 'LDN', 'Americas International Desk']
        all_data = []
        stats = {
            'total_rows': 0,
            'emea_rows': 0,
            'status_filtered': 0,
            'healthcare_rows': 0,
            'non_healthcare_rows': 0,
            'by_sheet': {}
        }
        
        # Process each sheet
        for sheet_name in sheet_names:
            try:
                # Read the specific sheet
                df_sheet = pd.read_excel(uploaded, sheet_name=sheet_name, engine='openpyxl')
                initial_rows = len(df_sheet)
                df_sheet['Source_Sheet'] = sheet_name
                
                # Validate the sheet has required columns
                required_cols = ['PU CTRY', 'STATUS', 'POD DATE/TIME']
                missing_cols = [col for col in required_cols if col not in df_sheet.columns]
                if missing_cols:
                    st.warning(f"Sheet {sheet_name} missing columns: {missing_cols}")
                
                # Clean and standardize PU CTRY
                if 'PU CTRY' in df_sheet.columns:
                    # Remove any trailing spaces and convert to uppercase for comparison
                    df_sheet['PU CTRY'] = df_sheet['PU CTRY'].astype(str).str.strip().str.upper()
                    # Filter EMEA countries
                    df_sheet_emea = df_sheet[df_sheet['PU CTRY'].isin(EMEA_COUNTRIES)]
                else:
                    df_sheet_emea = df_sheet
                
                emea_rows = len(df_sheet_emea)
                
                # Filter STATUS = 440-BILLED
                if 'STATUS' in df_sheet_emea.columns:
                    df_sheet_final = df_sheet_emea[df_sheet_emea['STATUS'].astype(str).str.strip() == '440-BILLED']
                else:
                    df_sheet_final = df_sheet_emea
                
                final_rows = len(df_sheet_final)
                
                stats['by_sheet'][sheet_name] = {
                    'initial': initial_rows,
                    'emea': emea_rows,
                    'final': final_rows
                }
                
                # Add to combined data only if there are rows
                if len(df_sheet_final) > 0:
                    all_data.append(df_sheet_final)
                    
            except Exception as e:
                st.warning(f"Could not read sheet {sheet_name}: {str(e)}")
                stats['by_sheet'][sheet_name] = {
                    'initial': 0,
                    'emea': 0,
                    'final': 0
                }
        
        # Combine all sheets
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
        else:
            combined_df = pd.DataFrame()
        
        stats['total_rows'] = sum(s['initial'] for s in stats['by_sheet'].values())
        stats['emea_rows'] = sum(s['emea'] for s in stats['by_sheet'].values())
        stats['status_filtered'] = len(combined_df)
        
        # Categorize into healthcare and non-healthcare
        healthcare_df = pd.DataFrame()
        non_healthcare_df = pd.DataFrame()
        
        if not combined_df.empty and 'ACCT NM' in combined_df.columns:
            # Apply healthcare classification with sheet context
            combined_df['Is_Healthcare'] = combined_df.apply(
                lambda row: is_healthcare(row.get('ACCT NM', ''), row.get('Source_Sheet', '')), axis=1
            )
            
            healthcare_df = combined_df[combined_df['Is_Healthcare'] == True].copy()
            non_healthcare_df = combined_df[combined_df['Is_Healthcare'] == False].copy()
            
            stats['healthcare_rows'] = len(healthcare_df)
            stats['non_healthcare_rows'] = len(non_healthcare_df)
        
        return healthcare_df, non_healthcare_df, stats
    
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), {}

# ---------------- Prep Functions ----------------
@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Each row = one entry (no dedup).
    Filters already applied in read_and_combine_sheets.
    Grouping month: POD (Actual Delivery) → YYYY-MM.
    OTP:
      - Gross = POD ≤ target (UPD DEL → QDT)
      - Net   = Gross OR (Late & NOT controllable)  => Net ≥ Gross
    """
    if df.empty:
        return df
    
    d = df.copy()
    
    # Parse POD dates - this is critical for monthly grouping
    if 'POD DATE/TIME' in d.columns:
        d["_pod"] = _excel_to_dt(d["POD DATE/TIME"])
        # Log how many valid POD dates we have
        valid_pods = d["_pod"].notna().sum()
        total_rows = len(d)
        if valid_pods < total_rows * 0.5:  # If less than 50% have valid POD
            st.warning(f"Only {valid_pods} out of {total_rows} rows have valid POD dates")
    else:
        d["_pod"] = pd.NaT
        st.error("POD DATE/TIME column not found!")
    
    # Parse target dates
    target_raw = _get_target_series(d)
    d["_target"] = _excel_to_dt(target_raw) if target_raw is not None else pd.NaT

    # Create month keys from POD - THIS IS THE KEY GROUPING
    # Each POD date gets grouped into its month
    d["Month_YYYY_MM"] = d["_pod"].dt.to_period("M").astype(str)  # e.g., '2025-01'
    d["Month_Sort"] = pd.to_datetime(d["Month_YYYY_MM"] + "-01", errors='coerce')
    d["Month_Display"] = d["Month_Sort"].dt.strftime("%b %Y")  # e.g., 'Jan 2025'

    # Controllable flag (QC NAME)
    if "QC NAME" in d.columns:
        d["QC_NAME_CLEAN"] = d["QC NAME"].astype(str)
        d["Is_Controllable"] = d["QC_NAME_CLEAN"].str.contains(CTRL_REGEX, na=False)
    else:
        d["QC_NAME_CLEAN"] = ""
        d["Is_Controllable"] = False

    # PIECES numeric
    if "PIECES" in d.columns:
        d["PIECES"] = pd.to_numeric(d["PIECES"], errors="coerce").fillna(0)
    else:
        d["PIECES"] = 0
    
    # TOTAL CHARGES numeric
    if "TOTAL CHARGES" in d.columns:
        d["TOTAL CHARGES"] = pd.to_numeric(d["TOTAL CHARGES"], errors="coerce").fillna(0)
    else:
        d["TOTAL CHARGES"] = 0

    # Row-level OTP calculation
    ok = d["_pod"].notna() & d["_target"].notna()
    d["On_Time_Gross"] = False
    d.loc[ok, "On_Time_Gross"] = d.loc[ok, "_pod"] <= d.loc[ok, "_target"]
    d["Late"] = ~d["On_Time_Gross"]
    d["On_Time_Net"] = d["On_Time_Gross"] | (d["Late"] & ~d["Is_Controllable"])

    return d

@st.cache_data(show_spinner=False)
def monthly_frames(d: pd.DataFrame):
    """Build Monthly OTP, Volume, Pieces, Revenue — all by POD month with the same keys."""
    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Filter to only rows with valid POD dates
    base_vol = d.dropna(subset=["_pod"]).copy()
    
    if base_vol.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Group by POD month for volume
    vol = (base_vol.groupby(["Month_YYYY_MM","Month_Display","Month_Sort"], as_index=False)
                 .size().rename(columns={"size":"Volume"}))

    # Group by POD month for pieces
    pieces = (base_vol.groupby(["Month_YYYY_MM","Month_Display","Month_Sort"], as_index=False)
                      .agg(Pieces=("PIECES","sum")))
    
    # Group by POD month for revenue
    revenue = (base_vol.groupby(["Month_YYYY_MM","Month_Display","Month_Sort"], as_index=False)
                       .agg(Revenue=("TOTAL CHARGES","sum")))

    # For OTP, need both POD and target dates
    base_otp = d.dropna(subset=["_pod","_target"]).copy()
    if base_otp.empty:
        otp = pd.DataFrame(columns=["Month_YYYY_MM","Month_Display","Month_Sort","Gross_OTP","Net_OTP"])
    else:
        # Group by POD month for OTP calculations
        otp = (base_otp.groupby(["Month_YYYY_MM","Month_Display","Month_Sort"], as_index=False)
                      .agg(Gross_On=("On_Time_Gross","sum"),
                           Gross_Tot=("On_Time_Gross","count"),
                           Net_On=("On_Time_Net","sum"),
                           Net_Tot=("On_Time_Net","count")))
        otp["Gross_OTP"] = (otp["Gross_On"] / otp["Gross_Tot"] * 100).round(2)
        otp["Net_OTP"]   = (otp["Net_On"]   / otp["Net_Tot"]   * 100).round(2)

    # Sort all dataframes by month
    vol, pieces, otp, revenue = [x.sort_values("Month_Sort") for x in (vol, pieces, otp, revenue)]
    return vol, pieces, otp, revenue

def calc_summary(d: pd.DataFrame):
    """Calculate summary statistics."""
    if d.empty:
        return np.nan, np.nan, 0, 0, 0, 0, 0
    
    base_otp = d.dropna(subset=["_pod","_target"])
    gross = base_otp["On_Time_Gross"].mean()*100 if len(base_otp) else np.nan
    net   = base_otp["On_Time_Net"].mean()*100   if len(base_otp) else np.nan
    if pd.notna(gross) and pd.notna(net) and net < gross:
        net = gross
    late_df         = base_otp[base_otp["Late"]]
    exceptions      = int(len(late_df))
    controllables   = int(late_df["Is_Controllable"].sum())
    uncontrollables = exceptions - controllables
    volume_total    = int(len(d.dropna(subset=["_pod"])))
    total_revenue   = float(d["TOTAL CHARGES"].sum()) if "TOTAL CHARGES" in d.columns else 0
    return (round(gross,2) if pd.notna(gross) else np.nan,
            round(net,2)   if pd.notna(net)   else np.nan,
            volume_total, exceptions, controllables, uncontrollables, total_revenue)

def create_dashboard_view(df: pd.DataFrame, tab_name: str, otp_target: float, debug_mode: bool = False):
    """Create dashboard view for a specific dataframe (healthcare or non-healthcare)."""
    if df.empty:
        st.info(f"No {tab_name} data available after filtering for EMEA countries and 440-BILLED status.")
        return
    
    # Process data
    processed_df = preprocess(df)
    
    if processed_df.empty:
        st.error(f"No {tab_name} data to display after processing.")
        return
    
    # Debug: Show monthly grouping details
    if debug_mode:
        with st.expander(f"🔍 Debug: {tab_name} Monthly Grouping"):
            pod_dates = processed_df.dropna(subset=["_pod"])
            if not pod_dates.empty:
                st.write(f"**Total rows processed:** {len(processed_df):,}")
                st.write(f"**Rows with valid POD dates:** {len(pod_dates):,}")
                
                # Show source sheet contribution
                if 'Source_Sheet' in pod_dates.columns:
                    st.write("\n**POD rows by source sheet:**")
                    source_counts = pod_dates['Source_Sheet'].value_counts()
                    for sheet, count in source_counts.items():
                        st.write(f"   {sheet}: {count:,} rows")
                
                # Monthly breakdown
                month_counts = pod_dates.groupby('Month_Display').size().sort_index()
                st.write("\n**Entries per month:**")
                st.dataframe(month_counts)
                
                # Total volume check
                st.write(f"\n**Total volume (sum of all months):** {month_counts.sum():,}")
                st.write(f"**Should equal rows with POD:** {len(pod_dates):,}")
                if month_counts.sum() == len(pod_dates):
                    st.success("✅ Monthly volumes add up correctly!")
                else:
                    st.error("❌ Monthly volume mismatch!")
    
    vol_pod, pieces_pod, otp_pod, revenue_pod = monthly_frames(processed_df)
    gross_otp, net_otp, volume_total, exceptions, controllables, uncontrollables, total_revenue = calc_summary(processed_df)
    
    # ---------------- KPIs & Gauges ----------------
    left, right = st.columns([1, 1.5])
    with left:
        st.markdown(f'<div class="kpi"><div class="k-num">{volume_total:,}</div><div class="k-cap">Volume (rows with POD)</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi"><div class="k-num">{_revenue_fmt(total_revenue)}</div><div class="k-cap">Total Revenue</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi"><div class="k-num">{exceptions:,}</div><div class="k-cap">Exceptions (Gross Late)</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi"><div class="k-num">{controllables:,}</div><div class="k-cap">Controllables (QC)</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi"><div class="k-num">{uncontrollables:,}</div><div class="k-cap">Uncontrollables (QC)</div></div>', unsafe_allow_html=True)

    with right:
        c1, c2, c3 = st.columns(3)
        with c1: 
            adjusted_otp = max(gross_otp, net_otp) if pd.notna(gross_otp) and pd.notna(net_otp) else (net_otp if pd.notna(net_otp) else gross_otp)
            st.plotly_chart(make_semi_gauge("Adjusted OTP", adjusted_otp),
                           use_container_width=True, config={"displayModeBar": False})
        with c2: 
            st.plotly_chart(make_semi_gauge("Controllable OTP", net_otp),
                           use_container_width=True, config={"displayModeBar": False})
        with c3: 
            st.plotly_chart(make_semi_gauge("Raw OTP", gross_otp),
                           use_container_width=True, config={"displayModeBar": False})

    with st.expander("Month & logic"):
        st.markdown(f"""
    **{tab_name} Data Processing:**
    - **Each row = one entry** (no shipment deduplication).
    - **Filter**: EMEA countries only, STATUS = 440-BILLED
    - Month basis: **POD DATE/TIME → YYYY-MM** (e.g., `2025-03-17 00:55` → `2025-03`)
    - **Volume** = number of rows with a POD in the month.
    - **Pieces** = sum(`PIECES`) across those rows in the month.
    - **Revenue** = sum(`TOTAL CHARGES`) across those rows in the month.
    - **Gross OTP** = `POD ≤ target (UPD DEL → QDT)`.
    - **Net/Adjusted OTP** = counts **non-controllable** lates as on-time (QC NAME contains Agent/Delivery agent/Customs/Warehouse) ⇒ **Net ≥ Gross**.
    """)

    st.markdown("---")

    # ---------------- Monthly Revenue Bar Chart ----------------
    st.subheader(f"💰 {tab_name}: Monthly Revenue Overview")
    
    if not revenue_pod.empty:
        revenue_sorted = revenue_pod.sort_values("Month_Sort")
        x = revenue_sorted["Month_Display"].tolist()
        y_revenue = revenue_sorted["Revenue"].astype(float).tolist()
        
        fig_rev = go.Figure()
        
        # Bar chart for monthly revenue
        fig_rev.add_trace(go.Bar(
            x=x,
            y=y_revenue,
            name="Monthly Revenue",
            marker_color=NAVY,
            text=[_revenue_fmt(v) for v in y_revenue],
            textposition="outside",
            textfont=dict(size=12, color=NAVY, family="Arial Black"),
            hovertemplate="<b>%{x}</b><br>Revenue: %{text}<extra></extra>"
        ))
        
        # Calculate and show average line
        avg_revenue = float(np.mean(y_revenue))
        fig_rev.add_shape(
            type="line",
            x0=-0.5,
            x1=len(x)-0.5,
            y0=avg_revenue,
            y1=avg_revenue,
            xref="x",
            yref="y",
            line=dict(color=GOLD, dash="dash", width=2)
        )
        
        fig_rev.add_annotation(
            x=len(x)-0.5,
            y=avg_revenue,
            xref="x",
            yref="y",
            text=f"Avg: {_revenue_fmt(avg_revenue)}",
            showarrow=False,
            xshift=-60,
            font=dict(size=11, color=GOLD),
            bgcolor="white",
            bordercolor=GOLD,
            borderwidth=1
        )
        
        # Add total revenue annotation
        total_rev = float(sum(y_revenue))
        fig_rev.add_annotation(
            x=0.5,
            y=1.05,
            xref="paper",
            yref="paper",
            text=f"<b>Total Revenue: {_revenue_fmt(total_rev)}</b>",
            showarrow=False,
            font=dict(size=14, color=NAVY),
            bgcolor="rgba(240, 180, 41, 0.1)",
            bordercolor=GOLD,
            borderwidth=2,
            borderpad=8
        )
        
        fig_rev.update_layout(
            height=500,
            plot_bgcolor="white",
            margin=dict(l=60, r=60, t=80, b=100),
            xaxis=dict(
                title="Month",
                tickangle=-30,
                tickfont=dict(size=11)
            ),
            yaxis=dict(
                title="Revenue ($)",
                titlefont=dict(size=13),
                tickfont=dict(size=11),
                gridcolor=GRID,
                showgrid=True
            ),
            showlegend=False
        )
        
        st.plotly_chart(fig_rev, use_container_width=True)
    else:
        st.info(f"No monthly revenue data available for {tab_name}")

    st.markdown("---")
