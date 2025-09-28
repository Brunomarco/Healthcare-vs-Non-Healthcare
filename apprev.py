import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import plotly.express as px

# ---------------- Page & Style ----------------
st.set_page_config(page_title="Radiopharma OTP Dashboard", page_icon="📊",
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
</style>
""", unsafe_allow_html=True)

st.title("Radiopharma OTP Dashboard")

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

# ---------------- Helpers ----------------
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
        return f"${n/1000000:.1f}M"
    elif n >= 1000:
        return f"${n/1000:.1f}K"
    else:
        return f"${n:.0f}"

def is_healthcare(account_name, sheet_name=None):
    """Determine if an account is healthcare-related."""
    if not account_name:
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
    """Create top/worst performance tables for a specific month"""
    if monthly_changes.empty:
        return
    
    month_data = monthly_changes[monthly_changes['Month'] == month].copy()
    if month_data.empty:
        st.info(f"No data available for {month}")
        return
    
    # Remove rows with NaN changes (first month for each account)
    month_data = month_data.dropna(subset=['Revenue_Change'])
    
    if month_data.empty:
        st.info(f"No month-over-month data available for {month}")
        return
    
    st.subheader(f"📊 {sector} Performance Analysis - {month}")
    
    # Create three columns for the metrics
    col1, col2, col3 = st.columns(3)
    
    # Revenue Performance
    with col1:
        st.markdown("### 💰 Revenue Performance")
        
        # Top 10 Revenue Increases
        top_revenue = month_data.nlargest(10, 'Revenue_Change')[
            ['Account', 'Revenue', 'Revenue_Prev', 'Revenue_Change', 'Revenue_Change_Pct']
        ].copy()
        
        if not top_revenue.empty:
            st.markdown("**🔝 Top 10 Revenue Increases**")
            # Format for display
            top_revenue['Change'] = top_revenue.apply(
                lambda x: f"+{_revenue_fmt(x['Revenue_Change'])} ({x['Revenue_Change_Pct']:.1f}%)" 
                if pd.notna(x['Revenue_Change_Pct']) else f"+{_revenue_fmt(x['Revenue_Change'])}", axis=1
            )
            top_revenue['Current'] = top_revenue['Revenue'].apply(_revenue_fmt)
            
            # Create styled dataframe
            st.dataframe(
                top_revenue[['Account', 'Current', 'Change']].head(10),
                use_container_width=True,
                hide_index=True
            )
        
        # Worst 10 Revenue Decreases
        worst_revenue = month_data.nsmallest(10, 'Revenue_Change')[
            ['Account', 'Revenue', 'Revenue_Prev', 'Revenue_Change', 'Revenue_Change_Pct']
        ].copy()
        
        if not worst_revenue.empty:
            st.markdown("**📉 Top 10 Revenue Decreases**")
            worst_revenue['Change'] = worst_revenue.apply(
                lambda x: f"{_revenue_fmt(x['Revenue_Change'])} ({x['Revenue_Change_Pct']:.1f}%)" 
                if pd.notna(x['Revenue_Change_Pct']) else f"{_revenue_fmt(x['Revenue_Change'])}", axis=1
            )
            worst_revenue['Current'] = worst_revenue['Revenue'].apply(_revenue_fmt)
            
            st.dataframe(
                worst_revenue[['Account', 'Current', 'Change']].head(10),
                use_container_width=True,
                hide_index=True
            )
    
    # Volume Performance
    with col2:
        st.markdown("### 📦 Volume Performance")
        
        # Top 10 Volume Increases
        volume_data = month_data.dropna(subset=['Volume_Change'])
        if not volume_data.empty:
            top_volume = volume_data.nlargest(10, 'Volume_Change')[
                ['Account', 'Volume', 'Volume_Change', 'Volume_Change_Pct']
            ].copy()
            
            st.markdown("**🔝 Top 10 Volume Increases**")
            top_volume['Change'] = top_volume.apply(
                lambda x: f"+{int(x['Volume_Change'])} ({x['Volume_Change_Pct']:.1f}%)" 
                if pd.notna(x['Volume_Change_Pct']) else f"+{int(x['Volume_Change'])}", axis=1
            )
            
            st.dataframe(
                top_volume[['Account', 'Volume', 'Change']].head(10),
                use_container_width=True,
                hide_index=True
            )
            
            # Worst 10 Volume Decreases
            worst_volume = volume_data.nsmallest(10, 'Volume_Change')[
                ['Account', 'Volume', 'Volume_Change', 'Volume_Change_Pct']
            ].copy()
            
            st.markdown("**📉 Top 10 Volume Decreases**")
            worst_volume['Change'] = worst_volume.apply(
                lambda x: f"{int(x['Volume_Change'])} ({x['Volume_Change_Pct']:.1f}%)" 
                if pd.notna(x['Volume_Change_Pct']) else f"{int(x['Volume_Change'])}", axis=1
            )
            
            st.dataframe(
                worst_volume[['Account', 'Volume', 'Change']].head(10),
                use_container_width=True,
                hide_index=True
            )
    
    # Pieces Performance
    with col3:
        st.markdown("### 📋 Pieces Performance")
        
        # Top 10 Pieces Increases
        pieces_data = month_data.dropna(subset=['Pieces_Change'])
        if not pieces_data.empty:
            top_pieces = pieces_data.nlargest(10, 'Pieces_Change')[
                ['Account', 'Pieces', 'Pieces_Change', 'Pieces_Change_Pct']
            ].copy()
            
            st.markdown("**🔝 Top 10 Pieces Increases**")
            top_pieces['Change'] = top_pieces.apply(
                lambda x: f"+{int(x['Pieces_Change'])} ({x['Pieces_Change_Pct']:.1f}%)" 
                if pd.notna(x['Pieces_Change_Pct']) else f"+{int(x['Pieces_Change'])}", axis=1
            )
            top_pieces['Pieces'] = top_pieces['Pieces'].astype(int)
            
            st.dataframe(
                top_pieces[['Account', 'Pieces', 'Change']].head(10),
                use_container_width=True,
                hide_index=True
            )
            
            # Worst 10 Pieces Decreases
            worst_pieces = pieces_data.nsmallest(10, 'Pieces_Change')[
                ['Account', 'Pieces', 'Pieces_Change', 'Pieces_Change_Pct']
            ].copy()
            
            st.markdown("**📉 Top 10 Pieces Decreases**")
            worst_pieces['Change'] = worst_pieces.apply(
                lambda x: f"{int(x['Pieces_Change'])} ({x['Pieces_Change_Pct']:.1f}%)" 
                if pd.notna(x['Pieces_Change_Pct']) else f"{int(x['Pieces_Change'])}", axis=1
            )
            worst_pieces['Pieces'] = worst_pieces['Pieces'].astype(int)
            
            st.dataframe(
                worst_pieces[['Account', 'Pieces', 'Change']].head(10),
                use_container_width=True,
                hide_index=True
            )
