#Code 1 - MBB Professional Edition - Marken Healthcare Logistics Dashboard
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import plotly.express as px
import os
import base64
import io
from PIL import Image

# ---------------- Page & Style ----------------
st.set_page_config(page_title="Marken Logistics Dashboard", page_icon="🟢",
                   layout="wide", initial_sidebar_state="collapsed")

# Define Marken brand colors - MBB Professional Palette
MARKEN_NAVY = "#003865"      # Marken primary navy blue
MARKEN_GREEN = "#8DC63F"     # Marken green
MARKEN_LIGHT_BLUE = "#0075BE"  # Secondary blue
MARKEN_GRAY = "#58595B"      # Text gray
MARKEN_LIGHT_GRAY = "#F8F9FA"  # Background gray (softer)

# MBB-Style Chart Color Palette
NAVY  = MARKEN_NAVY          
GOLD  = MARKEN_GREEN         
BLUE  = MARKEN_LIGHT_BLUE    
GREEN = MARKEN_GREEN         
SLATE = MARKEN_GRAY
GRID  = "#E9ECEF"            # Softer grid
RED   = "#C53030"            # Professional red
EMERALD = MARKEN_GREEN

# MBB Professional CSS - Executive Dashboard Style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&family=DM+Sans:wght@400;500;600;700&display=swap');

/* Global Reset & Base */
.main {
    padding: 0rem 2rem; 
    font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #FAFBFC;
}

/* Executive Headers */
h1 {
    color: #003865;
    font-weight: 700;
    font-family: 'DM Sans', sans-serif;
    font-size: 2.2rem;
    letter-spacing: -0.5px;
    border-bottom: none;
    padding-bottom: 0;
    margin-bottom: 0.5rem;
}

h2 {
    color: #003865;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    font-size: 1.5rem;
    margin-top: 2rem;
    margin-bottom: 1rem;
    letter-spacing: -0.3px;
}

h3 {
    color: #003865;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    font-size: 1.1rem;
    letter-spacing: -0.2px;
}

h4 {
    color: #003865;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
}

/* MBB-Style KPI Cards */
.kpi {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 4px;
    padding: 20px 24px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    transition: all 0.2s ease;
    margin-bottom: 12px;
}

.kpi:hover {
    border-color: #8DC63F;
    box-shadow: 0 2px 8px rgba(0,56,101,0.08);
}

.k-num {
    font-size: 32px;
    font-weight: 700;
    color: #003865;
    line-height: 1.1;
    font-family: 'DM Sans', sans-serif;
    letter-spacing: -1px;
}

.k-cap {
    font-size: 12px;
    color: #6B7280;
    margin-top: 6px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-family: 'Source Sans Pro', sans-serif;
}

/* Executive Summary Box */
.exec-summary {
    background: linear-gradient(135deg, #003865 0%, #004d80 100%);
    border-radius: 6px;
    padding: 24px 28px;
    color: white;
    margin-bottom: 24px;
}

.exec-summary h3 {
    color: white;
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 12px;
    opacity: 0.9;
}

.exec-summary .highlight {
    font-size: 42px;
    font-weight: 700;
    font-family: 'DM Sans', sans-serif;
    letter-spacing: -1px;
}

/* Professional Tab Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px; 
    background-color: transparent;
    padding: 0;
    border-bottom: 2px solid #E5E7EB;
}

.stTabs [data-baseweb="tab"] {
    height: 48px;
    padding: 0 28px;
    background-color: transparent;
    border-radius: 0;
    font-weight: 600;
    font-size: 14px;
    color: #6B7280;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    font-family: 'Source Sans Pro', sans-serif;
    letter-spacing: 0.3px;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #003865;
    background-color: transparent;
}

.stTabs [aria-selected="true"] {
    background-color: transparent;
    color: #003865;
    border-bottom: 2px solid #8DC63F;
    font-weight: 700;
}

/* Clean Data Tables - MBB Style */
.dataframe {
    font-size: 13px !important; 
    font-family: 'Source Sans Pro', sans-serif !important;
    border: none !important;
}

.dataframe td {
    padding: 12px 16px !important; 
    border-bottom: 1px solid #F3F4F6 !important;
    color: #374151 !important;
}

.dataframe th {
    padding: 14px 16px !important; 
    background-color: #F8F9FA !important; 
    color: #003865 !important; 
    font-weight: 600 !important;
    text-transform: uppercase !important;
    font-size: 11px !important;
    letter-spacing: 0.5px !important;
    border-bottom: 2px solid #E5E7EB !important;
}

.dataframe tr:hover td {
    background-color: #F8F9FA !important;
}

/* Professional Buttons */
.stButton>button {
    background-color: #003865; 
    color: white; 
    font-weight: 600; 
    border: none; 
    padding: 10px 24px; 
    border-radius: 4px; 
    transition: all 0.2s ease;
    font-family: 'Source Sans Pro', sans-serif;
    letter-spacing: 0.3px;
    font-size: 14px;
}

.stButton>button:hover {
    background-color: #004d80; 
    box-shadow: 0 4px 12px rgba(0,56,101,0.2);
}

/* Form Elements */
.stSelectbox label, .stTextInput label {
    color: #374151; 
    font-weight: 600;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Metrics Cards */
.stMetric label {
    color: #6B7280; 
    font-weight: 600; 
    text-transform: uppercase; 
    font-size: 11px;
    letter-spacing: 0.5px;
}

.stMetric [data-testid="metric-container"] {
    background-color: white; 
    padding: 16px 20px; 
    border-radius: 4px; 
    border: 1px solid #E5E7EB;
}

/* Expander Styling */
.streamlit-expanderHeader {
    font-family: 'Source Sans Pro', sans-serif;
    font-weight: 600;
    font-size: 14px;
    color: #374151;
    background-color: #F8F9FA;
    border-radius: 4px;
}

/* Info/Warning Boxes */
.stAlert {
    border-radius: 4px;
    border-left: 4px solid;
    font-family: 'Source Sans Pro', sans-serif;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E5E7EB;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

/* Custom Divider */
hr {
    border: none;
    height: 1px;
    background-color: #E5E7EB;
    margin: 2rem 0;
}

/* Insight Box - MBB Style */
.insight-box {
    background: #F0FDF4;
    border-left: 4px solid #8DC63F;
    padding: 16px 20px;
    border-radius: 0 4px 4px 0;
    margin: 16px 0;
}

.insight-box-warning {
    background: #FEF3C7;
    border-left: 4px solid #F59E0B;
    padding: 16px 20px;
    border-radius: 0 4px 4px 0;
    margin: 16px 0;
}

.insight-box-info {
    background: #EFF6FF;
    border-left: 4px solid #3B82F6;
    padding: 16px 20px;
    border-radius: 0 4px 4px 0;
    margin: 16px 0;
}

/* Section Header with Line */
.section-header {
    display: flex;
    align-items: center;
    margin: 2rem 0 1.5rem 0;
}

.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #E5E7EB;
    margin-left: 16px;
}

/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Function to load and display logo
def load_logo():
    """Load Marken logo if available"""
    logo_paths = [
        "/mnt/user-data/uploads/1762775253257_image.png",
        "/home/claude/marken_logo.png",
        "marken_logo.png"
    ]
    
    for logo_path in logo_paths:
        if os.path.exists(logo_path):
            return Image.open(logo_path)
    return None

def get_logo_base64():
    """Convert logo to base64 for inline display"""
    logo_image = load_logo()
    if logo_image:
        import io
        buffer = io.BytesIO()
        logo_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    return None

# Get logo as base64 for inline use
logo_base64 = get_logo_base64()

# Create inline logo HTML
def get_inline_logo_html(height=20):
    """Get HTML for inline logo display"""
    if logo_base64:
        return f'<img src="{logo_base64}" style="height:{height}px; vertical-align:middle; margin-right:5px;">'
    else:
        return '<span style="color:#8DC63F; font-weight:800;">⬢⬢⬢</span> <span style="color:#003865; font-weight:700;">MARKEN</span>'

# MBB Professional Header
logo_image = load_logo()

if logo_image:
    col1, col2, col3 = st.columns([2, 4, 2])
    
    with col1:
        st.image(logo_image, width=220)
    
    with col3:
        st.markdown("""
        <div style="text-align: right; padding-top: 15px;">
            <span style="color: #6B7280; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">OTP Performance Report</span><br>
            <span style="color: #003865; font-size: 15px; font-weight: 600;">Healthcare & Logistics Analytics</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 15px 0 25px 0;'>", unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="padding: 20px 0; border-bottom: 1px solid #E5E7EB; margin-bottom: 25px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="color: #8DC63F; font-size: 22px; font-weight: bold;">⬢⬢⬢</span>
                <span style="color: #003865; font-size: 28px; font-weight: 700; margin-left: 8px; font-family: 'DM Sans', sans-serif;">MARKEN</span>
                <span style="color: #6B7280; font-size: 13px; margin-left: 8px;">a UPS Company</span>
            </div>
            <div style="text-align: right;">
                <span style="color: #6B7280; font-size: 12px; text-transform: uppercase; letter-spacing: 1px;">OTP Performance Report</span><br>
                <span style="color: #003865; font-size: 15px; font-weight: 600;">Healthcare & Logistics Analytics</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main Title - MBB Style
st.markdown("""
<div style="text-align: center; margin-bottom: 30px;">
    <h1 style="margin-bottom: 8px;">Logistics Performance Dashboard</h1>
    <p style="color: #6B7280; font-size: 15px; margin: 0;">On-Time Performance & Operational Analytics | EMEA Region</p>
</div>
""", unsafe_allow_html=True)

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
    EXCLUDE_FROM_HEALTHCARE = {"avid", "lantheus", "life"}
    lower = str(account_name).strip().lower()
    if any(excluded in lower for excluded in EXCLUDE_FROM_HEALTHCARE):
        return False
      
    if sheet_name == 'AMS':
        return True
    if sheet_name == 'Aviation SVC':
        return False
    
    lower = str(account_name).lower()
    
    for keyword in NON_HEALTHCARE_KEYWORDS:
        if keyword in lower:
            return False
    
    for keyword in HEALTHCARE_KEYWORDS:
        if keyword in lower:
            return True
    
    return False

# MBB-Style Gauge Chart
def make_semi_gauge(title: str, value: float) -> go.Figure:
    """MBB-style semi-donut gauge - clean and professional."""
    v = max(0.0, min(100.0, 0.0 if pd.isna(value) else float(value)))
    fig = go.Figure()
    
    # Determine color based on performance
    if v >= 95:
        gauge_color = "#10B981"  # Emerald green for good
    elif v >= 85:
        gauge_color = "#F59E0B"  # Amber for moderate
    else:
        gauge_color = "#EF4444"  # Red for needs attention
    
    fig.add_trace(go.Pie(
        values=[v, 100 - v, 100],
        hole=0.78, sort=False, direction="clockwise", rotation=180,
        textinfo="none", showlegend=False,
        marker=dict(colors=[gauge_color, "#F3F4F6", "rgba(0,0,0,0)"])
    ))
    
    # Center value
    fig.add_annotation(
        text=f"<b>{v:.1f}%</b>", 
        x=0.5, y=0.55, xref="paper", yref="paper",
        showarrow=False, 
        font=dict(size=26, color=MARKEN_NAVY, family="DM Sans, sans-serif")
    )
    
    # Title below - positioned lower with more space
    fig.add_annotation(
        text=f"<b>{title}</b>", 
        x=0.5, y=0.08, xref="paper", yref="paper",
        showarrow=False, 
        font=dict(size=13, color="#374151", family="Source Sans Pro, sans-serif")
    )
    
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=50), 
        height=220, 
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def analyze_monthly_changes(df: pd.DataFrame, metric: str = 'TOTAL CHARGES'):
    """Analyze month-over-month changes for accounts"""
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    
    monthly = df.groupby(['ACCT NM', 'Month_Display', 'Month_Sort']).agg({
        metric: 'sum',
        'PIECES': 'sum' if 'PIECES' in df.columns else 'size',
        '_pod': 'count'
    }).reset_index()
    monthly.columns = ['Account', 'Month', 'Month_Sort', 'Revenue', 'Pieces', 'Volume']
    
    monthly = monthly.sort_values(['Account', 'Month_Sort'])
    
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
    """Create MBB-style performance tables for a specific month"""
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
    
    # Section header
    st.markdown(f"""
    <div style="margin: 36px 0 20px 0;">
        <span style="color: #003865; font-size: 18px; font-weight: 600; font-family: 'DM Sans', sans-serif;">
            Performance Analysis — {month}
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    # Revenue Performance
    with col1:
        st.markdown("""
        <div style="color: #003865; font-size: 14px; font-weight: 600; margin-bottom: 12px; 
                    border-bottom: 2px solid #8DC63F; padding-bottom: 8px;">
            💰 Revenue Performance
        </div>
        """, unsafe_allow_html=True)
        
        top_revenue = month_data.nlargest(10, 'Revenue_Change')[
            ['Account', 'Revenue', 'Revenue_Prev', 'Revenue_Change', 'Revenue_Change_Pct', 'Volume_Change', 'Volume_Change_Pct']
        ].copy()
        
        if not top_revenue.empty:
            st.markdown("<span style='color:#10B981; font-weight:600; font-size:13px;'>▲ Top 10 Increases</span>", unsafe_allow_html=True)
            top_revenue['Rev Change'] = top_revenue.apply(
                lambda x: f"+{_revenue_fmt(x['Revenue_Change'])} ({x['Revenue_Change_Pct']:.1f}%)" 
                if pd.notna(x['Revenue_Change_Pct']) else f"+{_revenue_fmt(x['Revenue_Change'])}", axis=1
            )
            top_revenue['Vol Δ'] = top_revenue.apply(
                lambda x: f"{int(x['Volume_Change']):+d}" 
                if pd.notna(x['Volume_Change']) else "—", axis=1
            )
            top_revenue['Current'] = top_revenue['Revenue'].apply(_revenue_fmt)
            
            st.dataframe(
                top_revenue[['Account', 'Current', 'Rev Change', 'Vol Δ']].head(10),
                use_container_width=True,
                hide_index=True,
                height=380
            )
        
        worst_revenue = month_data.nsmallest(10, 'Revenue_Change')[
            ['Account', 'Revenue', 'Revenue_Prev', 'Revenue_Change', 'Revenue_Change_Pct', 'Volume_Change', 'Volume_Change_Pct']
        ].copy()
        
        if not worst_revenue.empty:
            st.markdown("<span style='color:#EF4444; font-weight:600; font-size:13px;'>▼ Top 10 Decreases</span>", unsafe_allow_html=True)
            worst_revenue['Rev Change'] = worst_revenue.apply(
                lambda x: f"{_revenue_fmt(x['Revenue_Change'])} ({x['Revenue_Change_Pct']:.1f}%)" 
                if pd.notna(x['Revenue_Change_Pct']) else f"{_revenue_fmt(x['Revenue_Change'])}", axis=1
            )
            worst_revenue['Vol Δ'] = worst_revenue.apply(
                lambda x: f"{int(x['Volume_Change']):+d}" 
                if pd.notna(x['Volume_Change']) else "—", axis=1
            )
            worst_revenue['Current'] = worst_revenue['Revenue'].apply(_revenue_fmt)
            
            st.dataframe(
                worst_revenue[['Account', 'Current', 'Rev Change', 'Vol Δ']].head(10),
                use_container_width=True,
                hide_index=True,
                height=380
            )
    
    # Volume Performance
    with col2:
        st.markdown("""
        <div style="color: #003865; font-size: 14px; font-weight: 600; margin-bottom: 12px; 
                    border-bottom: 2px solid #0075BE; padding-bottom: 8px;">
            📦 Volume Performance
        </div>
        """, unsafe_allow_html=True)
        
        volume_data = month_data.dropna(subset=['Volume_Change'])
        if not volume_data.empty:
            top_volume = volume_data.nlargest(10, 'Volume_Change')[
                ['Account', 'Volume', 'Volume_Change', 'Volume_Change_Pct', 'Revenue_Change', 'Revenue_Change_Pct']
            ].copy()
            
            st.markdown("<span style='color:#10B981; font-weight:600; font-size:13px;'>▲ Top 10 Increases</span>", unsafe_allow_html=True)
            top_volume['Vol Change'] = top_volume.apply(
                lambda x: f"+{int(x['Volume_Change'])} ({x['Volume_Change_Pct']:.1f}%)" 
                if pd.notna(x['Volume_Change_Pct']) else f"+{int(x['Volume_Change'])}", axis=1
            )
            top_volume['Rev Δ'] = top_volume.apply(
                lambda x: f"{_revenue_fmt(x['Revenue_Change'])}" 
                if pd.notna(x['Revenue_Change']) else "—", axis=1
            )
            
            st.dataframe(
                top_volume[['Account', 'Volume', 'Vol Change', 'Rev Δ']].head(10),
                use_container_width=True,
                hide_index=True,
                height=380
            )
            
            worst_volume = volume_data.nsmallest(10, 'Volume_Change')[
                ['Account', 'Volume', 'Volume_Change', 'Volume_Change_Pct', 'Revenue_Change', 'Revenue_Change_Pct']
            ].copy()
            
            st.markdown("<span style='color:#EF4444; font-weight:600; font-size:13px;'>▼ Top 10 Decreases</span>", unsafe_allow_html=True)
            worst_volume['Vol Change'] = worst_volume.apply(
                lambda x: f"{int(x['Volume_Change'])} ({x['Volume_Change_Pct']:.1f}%)" 
                if pd.notna(x['Volume_Change_Pct']) else f"{int(x['Volume_Change'])}", axis=1
            )
            worst_volume['Rev Δ'] = worst_volume.apply(
                lambda x: f"{_revenue_fmt(x['Revenue_Change'])}" 
                if pd.notna(x['Revenue_Change']) else "—", axis=1
            )
            
            st.dataframe(
                worst_volume[['Account', 'Volume', 'Vol Change', 'Rev Δ']].head(10),
                use_container_width=True,
                hide_index=True,
                height=380
            )
    
    # Pieces Performance
    with col3:
        st.markdown("""
        <div style="color: #003865; font-size: 14px; font-weight: 600; margin-bottom: 12px; 
                    border-bottom: 2px solid #6B7280; padding-bottom: 8px;">
            📋 Pieces Performance
        </div>
        """, unsafe_allow_html=True)
        
        pieces_data = month_data.dropna(subset=['Pieces_Change'])
        if not pieces_data.empty:
            top_pieces = pieces_data.nlargest(10, 'Pieces_Change')[
                ['Account', 'Pieces', 'Pieces_Change', 'Pieces_Change_Pct']
            ].copy()
            
            st.markdown("<span style='color:#10B981; font-weight:600; font-size:13px;'>▲ Top 10 Increases</span>", unsafe_allow_html=True)
            top_pieces['Change'] = top_pieces.apply(
                lambda x: f"+{int(x['Pieces_Change'])} ({x['Pieces_Change_Pct']:.1f}%)" 
                if pd.notna(x['Pieces_Change_Pct']) else f"+{int(x['Pieces_Change'])}", axis=1
            )
            top_pieces['Pieces'] = top_pieces['Pieces'].astype(int)
            
            st.dataframe(
                top_pieces[['Account', 'Pieces', 'Change']].head(10),
                use_container_width=True,
                hide_index=True,
                height=380
            )
            
            worst_pieces = pieces_data.nsmallest(10, 'Pieces_Change')[
                ['Account', 'Pieces', 'Pieces_Change', 'Pieces_Change_Pct']
            ].copy()
            
            st.markdown("<span style='color:#EF4444; font-weight:600; font-size:13px;'>▼ Top 10 Decreases</span>", unsafe_allow_html=True)
            worst_pieces['Change'] = worst_pieces.apply(
                lambda x: f"{int(x['Pieces_Change'])} ({x['Pieces_Change_Pct']:.1f}%)" 
                if pd.notna(x['Pieces_Change_Pct']) else f"{int(x['Pieces_Change'])}", axis=1
            )
            worst_pieces['Pieces'] = worst_pieces['Pieces'].astype(int)
            
            st.dataframe(
                worst_pieces[['Account', 'Pieces', 'Change']].head(10),
                use_container_width=True,
                hide_index=True,
                height=380
            )

# ---------------- IO - CRITICAL FUNCTION ----------------
@st.cache_data(show_spinner=False)
def read_and_combine_sheets(uploaded):
    """Read ALL rows from ALL sheets, then apply filters."""
    try:
        xl_file = pd.ExcelFile(uploaded, engine='openpyxl')
        all_sheet_names = xl_file.sheet_names
        
        all_data_raw = []
        all_data_filtered = []
        all_data_emea_only = []
        
        radiopharma_raw = []
        radiopharma_filtered = []
        radiopharma_emea_only = []
        
        stats = {
            'total_rows_raw': 0,
            'total_rows': 0,
            'emea_rows': 0,
            'status_filtered': 0,
            'healthcare_rows': 0,
            'non_healthcare_rows': 0,
            'radiopharma_rows': 0,
            'by_sheet': {},
            'sheets_read': all_sheet_names
        }
        
        for sheet_name in all_sheet_names:
            try:
                df_sheet_raw = pd.read_excel(uploaded, sheet_name=sheet_name, engine='openpyxl')
                
                if df_sheet_raw.empty or len(df_sheet_raw) == 0:
                    stats['by_sheet'][sheet_name] = {
                        'raw_rows': 0,
                        'initial': 0,
                        'emea': 0,
                        'final': 0,
                        'note': 'Empty sheet'
                    }
                    continue
                
                raw_row_count = len(df_sheet_raw)
                stats['total_rows_raw'] += raw_row_count
                
                df_sheet_raw['Source_Sheet'] = sheet_name
                
                if sheet_name == 'RadioPharma':
                    radiopharma_raw.append(df_sheet_raw.copy())
                else:
                    all_data_raw.append(df_sheet_raw.copy())
                
                df_sheet = df_sheet_raw.copy()
                initial_rows = len(df_sheet)
                stats['total_rows'] += initial_rows
                
                if 'PU CTRY' in df_sheet.columns:
                    df_sheet['PU CTRY'] = df_sheet['PU CTRY'].astype(str).str.strip().str.upper()
                    df_sheet['PU CTRY'] = df_sheet['PU CTRY'].replace(['NAN', 'NONE', '<NA>'], '')
                    df_sheet_emea = df_sheet[
                        (df_sheet['PU CTRY'].isin(EMEA_COUNTRIES)) | 
                        (df_sheet['PU CTRY'] == '') |
                        (df_sheet['PU CTRY'].isna())
                    ]
                else:
                    df_sheet_emea = df_sheet
                
                emea_rows = len(df_sheet_emea)
                
                if len(df_sheet_emea) > 0:
                    if sheet_name == 'RadioPharma':
                        radiopharma_emea_only.append(df_sheet_emea.copy())
                    else:
                        all_data_emea_only.append(df_sheet_emea.copy())
                
                if 'STATUS' in df_sheet_emea.columns:
                    df_sheet_emea['STATUS'] = df_sheet_emea['STATUS'].astype(str).str.strip()
                    df_sheet_final = df_sheet_emea[
                        (df_sheet_emea['STATUS'] == '440-BILLED') |
                        (df_sheet_emea['STATUS'] == '') |
                        (df_sheet_emea['STATUS'] == 'nan') |
                        (df_sheet_emea['STATUS'].isna())
                    ]
                else:
                    df_sheet_final = df_sheet_emea
                
                final_rows = len(df_sheet_final)
                
                stats['by_sheet'][sheet_name] = {
                    'raw_rows': raw_row_count,
                    'initial': initial_rows,
                    'emea': emea_rows,
                    'final': final_rows
                }
                
                if len(df_sheet_final) > 0:
                    if sheet_name == 'RadioPharma':
                        radiopharma_filtered.append(df_sheet_final)
                    else:
                        all_data_filtered.append(df_sheet_final)
                    
            except Exception as e:
                st.warning(f"Could not read sheet '{sheet_name}': {str(e)}")
                stats['by_sheet'][sheet_name] = {
                    'raw_rows': 0,
                    'initial': 0,
                    'emea': 0,
                    'final': 0,
                    'error': str(e)
                }
        
        if all_data_raw:
            combined_raw_df = pd.concat(all_data_raw, ignore_index=True)
            stats['total_combined_raw'] = len(combined_raw_df)
        else:
            combined_raw_df = pd.DataFrame()
            stats['total_combined_raw'] = 0
        
        if radiopharma_raw:
            radiopharma_raw_df = pd.concat(radiopharma_raw, ignore_index=True)
        else:
            radiopharma_raw_df = pd.DataFrame()
        
        if all_data_emea_only:
            combined_emea_df = pd.concat(all_data_emea_only, ignore_index=True)
        else:
            combined_emea_df = pd.DataFrame()
        
        if radiopharma_emea_only:
            radiopharma_gross_df = pd.concat(radiopharma_emea_only, ignore_index=True)
        else:
            radiopharma_gross_df = pd.DataFrame()
        
        if all_data_filtered:
            combined_df = pd.concat(all_data_filtered, ignore_index=True)
        else:
            combined_df = pd.DataFrame()
        
        if radiopharma_filtered:
            radiopharma_df = pd.concat(radiopharma_filtered, ignore_index=True)
            stats['radiopharma_rows'] = len(radiopharma_df)
        else:
            radiopharma_df = pd.DataFrame()
            stats['radiopharma_rows'] = 0
        
        stats['emea_rows'] = sum(s.get('emea', 0) for s in stats['by_sheet'].values())
        stats['status_filtered'] = len(combined_df) + len(radiopharma_df)
        
        if stats['total_rows_raw'] > 0:
            retention_rate = (stats['status_filtered'] / stats['total_rows_raw']) * 100
            if retention_rate < 50:
                st.warning(f"⚠️ Only {retention_rate:.1f}% of raw data retained after filtering.")
        
        healthcare_df = pd.DataFrame()
        non_healthcare_df = pd.DataFrame()
        
        if not combined_df.empty and 'ACCT NM' in combined_df.columns:
            combined_df['Is_Healthcare'] = combined_df.apply(
                lambda row: is_healthcare(row.get('ACCT NM', ''), row.get('Source_Sheet', '')), axis=1
            )
            
            healthcare_df = combined_df[combined_df['Is_Healthcare'] == True].copy()
            non_healthcare_df = combined_df[combined_df['Is_Healthcare'] == False].copy()
            
            stats['healthcare_rows'] = len(healthcare_df)
            stats['non_healthcare_rows'] = len(non_healthcare_df)
        
        healthcare_gross_df = pd.DataFrame()
        non_healthcare_gross_df = pd.DataFrame()
        
        if not combined_emea_df.empty and 'ACCT NM' in combined_emea_df.columns:
            combined_emea_df['Is_Healthcare'] = combined_emea_df.apply(
                lambda row: is_healthcare(row.get('ACCT NM', ''), row.get('Source_Sheet', '')), axis=1
            )
            
            healthcare_gross_df = combined_emea_df[combined_emea_df['Is_Healthcare'] == True].copy()
            non_healthcare_gross_df = combined_emea_df[combined_emea_df['Is_Healthcare'] == False].copy()
        
        account_classification = pd.DataFrame()
        
        all_raw_for_classification = []
        if not combined_raw_df.empty:
            all_raw_for_classification.append(combined_raw_df)
        if not radiopharma_raw_df.empty:
            all_raw_for_classification.append(radiopharma_raw_df)
        
        if all_raw_for_classification:
            all_raw_combined = pd.concat(all_raw_for_classification, ignore_index=True)
            
            if 'ACCT NM' in all_raw_combined.columns:
                unique_accounts = all_raw_combined['ACCT NM'].dropna().unique()
                classifications = []
                
                for account in unique_accounts:
                    account_rows = all_raw_combined[all_raw_combined['ACCT NM'] == account]
                    is_emea = False
                    if 'PU CTRY' in account_rows.columns:
                        countries = account_rows['PU CTRY'].astype(str).str.strip().str.upper().unique()
                        is_emea = any(c in EMEA_COUNTRIES for c in countries if c not in ['NAN', 'NONE', '<NA>', ''])
                    
                    from_radiopharma = 'RadioPharma' in account_rows['Source_Sheet'].unique()
                    
                    if from_radiopharma:
                        if is_emea:
                            if not radiopharma_df.empty and account in radiopharma_df['ACCT NM'].values:
                                classification = 'RadioPharma'
                            else:
                                classification = 'RadioPharma (Not Used - Status Filter)'
                        else:
                            classification = 'Not Used (Non-EMEA RadioPharma)'
                    elif not is_emea:
                        classification = 'Not Used (Non-EMEA)'
                    else:
                        sheet_context = account_rows['Source_Sheet'].iloc[0] if 'Source_Sheet' in account_rows.columns else None
                        is_hc = is_healthcare(account, sheet_context)
                        
                        if is_hc:
                            classification = 'Healthcare'
                        else:
                            classification = 'Non-Healthcare'
                    
                    total_rows = len(account_rows)
                    
                    classifications.append({
                        'Account Name': account,
                        'Classification': classification,
                        'Total Rows': total_rows,
                        'Source Sheets': ', '.join(account_rows['Source_Sheet'].unique()) if 'Source_Sheet' in account_rows.columns else 'N/A'
                    })
                
                account_classification = pd.DataFrame(classifications)
        
        return healthcare_df, non_healthcare_df, stats, healthcare_gross_df, non_healthcare_gross_df, account_classification, radiopharma_df, radiopharma_gross_df
    
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame(), pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# ---------------- Prep ----------------
@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Each row = one entry (no dedup).
    """
    if df.empty:
        return df
    
    d = df.copy()
    
    if 'POD DATE/TIME' in d.columns:
        d["_pod"] = _excel_to_dt(d["POD DATE/TIME"])
        valid_pods = d["_pod"].notna().sum()
        total_rows = len(d)
        if valid_pods < total_rows * 0.5:
            st.warning(f"Only {valid_pods} out of {total_rows} rows have valid POD dates")
    else:
        d["_pod"] = pd.NaT
        st.error("POD DATE/TIME column not found!")
    
    target_raw = _get_target_series(d)
    d["_target"] = _excel_to_dt(target_raw) if target_raw is not None else pd.NaT

    d["Month_YYYY_MM"] = d["_pod"].dt.to_period("M").astype(str)
    d["Month_Sort"] = pd.to_datetime(d["Month_YYYY_MM"] + "-01", errors='coerce')
    d["Month_Display"] = d["Month_Sort"].dt.strftime("%b %Y")

    if "QC NAME" in d.columns:
        d["QC_NAME_CLEAN"] = d["QC NAME"].astype(str)
        d["Is_Controllable"] = d["QC_NAME_CLEAN"].str.contains(CTRL_REGEX, na=False)
    else:
        d["QC_NAME_CLEAN"] = ""
        d["Is_Controllable"] = False

    if "PIECES" in d.columns:
        d["PIECES"] = pd.to_numeric(d["PIECES"], errors="coerce").fillna(0)
    else:
        d["PIECES"] = 0
    
    if "TOTAL CHARGES" in d.columns:
        d["TOTAL CHARGES"] = pd.to_numeric(d["TOTAL CHARGES"], errors="coerce").fillna(0)
    else:
        d["TOTAL CHARGES"] = 0

    ok = d["_pod"].notna() & d["_target"].notna()
    d["On_Time_Gross"] = False
    d.loc[ok, "On_Time_Gross"] = d.loc[ok, "_pod"] <= d.loc[ok, "_target"]
    d["Late"] = ~d["On_Time_Gross"]
    d["On_Time_Net"] = d["On_Time_Gross"] | (d["Late"] & ~d["Is_Controllable"])

    return d

@st.cache_data(show_spinner=False)
def monthly_frames(d: pd.DataFrame):
    """Build Monthly OTP, Volume, Pieces, Revenue."""
    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    base_vol = d.dropna(subset=["_pod"]).copy()
    
    if base_vol.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    vol = (base_vol.groupby(["Month_YYYY_MM","Month_Display","Month_Sort"], as_index=False)
                 .size().rename(columns={"size":"Volume"}))

    pieces = (base_vol.groupby(["Month_YYYY_MM","Month_Display","Month_Sort"], as_index=False)
                      .agg(Pieces=("PIECES","sum")))
    
    revenue = (base_vol.groupby(["Month_YYYY_MM","Month_Display","Month_Sort"], as_index=False)
                       .agg(Revenue=("TOTAL CHARGES","sum")))

    base_otp = d.dropna(subset=["_pod","_target"]).copy()
    if base_otp.empty:
        otp = pd.DataFrame(columns=["Month_YYYY_MM","Month_Display","Month_Sort","Gross_OTP","Net_OTP"])
    else:
        otp = (base_otp.groupby(["Month_YYYY_MM","Month_Display","Month_Sort"], as_index=False)
                      .agg(Gross_On=("On_Time_Gross","sum"),
                           Gross_Tot=("On_Time_Gross","count"),
                           Net_On=("On_Time_Net","sum"),
                           Net_Tot=("On_Time_Net","count")))
        otp["Gross_OTP"] = (otp["Gross_On"] / otp["Gross_Tot"] * 100).round(2)
        otp["Net_OTP"]   = (otp["Net_On"]   / otp["Net_Tot"]   * 100).round(2)

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

# MBB-Style Chart Configuration
def get_mbb_chart_layout(title="", height=450, show_legend=True):
    """Return MBB-style chart layout configuration."""
    layout = dict(
        height=height,
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=60, t=80, b=80),
        xaxis=dict(
            tickfont=dict(size=11, color="#6B7280", family="Source Sans Pro, sans-serif"),
            tickangle=-30,
            showgrid=False,
            showline=True,
            linecolor="#E5E7EB",
            linewidth=1
        ),
        yaxis=dict(
            tickfont=dict(size=11, color="#6B7280", family="Source Sans Pro, sans-serif"),
            gridcolor="#F3F4F6",
            gridwidth=1,
            showgrid=True,
            zeroline=False,
            showline=True,
            linecolor="#E5E7EB",
            linewidth=1
        )
    )
    
    if title:
        layout['title'] = dict(
            text=title,
            font=dict(size=16, color=MARKEN_NAVY, family="DM Sans, sans-serif"),
            x=0,
            xanchor='left',
            y=0.95
        )
    
    if show_legend:
        layout['legend'] = dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            x=0.5,
            xanchor="center",
            font=dict(size=12, family="Source Sans Pro, sans-serif"),
            bgcolor="rgba(255,255,255,0.8)"
        )
    else:
        layout['showlegend'] = False
    
    return layout

def create_dashboard_view(df: pd.DataFrame, tab_name: str, otp_target: float, gross_df: pd.DataFrame = None, debug_mode: bool = False):
    """Create MBB-style dashboard view."""
    if df.empty:
        st.info(f"No {tab_name} data available after filtering for EMEA countries and 440-BILLED status.")
        return
    
    processed_df = preprocess(df)
    
    if processed_df.empty:
        st.error(f"No {tab_name} data to display after processing.")
        return
    
    if debug_mode:
        with st.expander(f"🔍 Debug: {tab_name} Monthly Grouping"):
            pod_dates = processed_df.dropna(subset=["_pod"])
            if not pod_dates.empty:
                st.write(f"**Total rows processed:** {len(processed_df):,}")
                st.write(f"**Rows with valid POD dates:** {len(pod_dates):,}")
                
                if 'Source_Sheet' in pod_dates.columns:
                    st.write("\n**POD rows by source sheet:**")
                    source_counts = pod_dates['Source_Sheet'].value_counts()
                    for sheet, count in source_counts.items():
                        st.write(f"   {sheet}: {count:,} rows")
                
                month_counts = pod_dates.groupby('Month_Display').size().sort_index()
                st.write("\n**Entries per month:**")
                st.dataframe(month_counts)
    
    vol_pod, pieces_pod, otp_pod, revenue_pod = monthly_frames(processed_df)
    gross_otp, net_otp, volume_total, exceptions, controllables, uncontrollables, total_revenue = calc_summary(processed_df)
    
    # ---------------- Executive Summary Section ----------------
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #003865 0%, #0056A0 100%); border-radius: 8px; padding: 28px 32px; margin-bottom: 28px;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <div style="color: rgba(255,255,255,0.7); font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px;">
                    Executive Summary
                </div>
                <div style="color: white; font-size: 42px; font-weight: 700; font-family: 'DM Sans', sans-serif; letter-spacing: -1px;">
                    {net_otp:.1f}% <span style="font-size: 18px; font-weight: 400; opacity: 0.8;">Controllable OTP</span>
                </div>
                <div style="color: rgba(255,255,255,0.8); font-size: 14px; margin-top: 8px;">
                    Target: {otp_target}% | {'✓ Above target' if net_otp >= otp_target else '⚠ Below target'} by {abs(net_otp - otp_target):.1f}pp
                </div>
            </div>
            <div style="text-align: right;">
                <div style="color: white; font-size: 28px; font-weight: 600;">{volume_total:,}</div>
                <div style="color: rgba(255,255,255,0.7); font-size: 12px; text-transform: uppercase;">Total Shipments</div>
                <div style="color: white; font-size: 28px; font-weight: 600; margin-top: 12px;">{_revenue_fmt(total_revenue)}</div>
                <div style="color: rgba(255,255,255,0.7); font-size: 12px; text-transform: uppercase;">Total Revenue</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ---------------- KPIs Row ----------------
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="kpi">
            <div class="k-num">{volume_total:,}</div>
            <div class="k-cap">Volume</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi">
            <div class="k-num">{_revenue_fmt(total_revenue)}</div>
            <div class="k-cap">Revenue</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="kpi">
            <div class="k-num">{exceptions:,}</div>
            <div class="k-cap">Exceptions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="kpi">
            <div class="k-num" style="color: #EF4444;">{controllables:,}</div>
            <div class="k-cap">Controllables</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="kpi">
            <div class="k-num" style="color: #6B7280;">{uncontrollables:,}</div>
            <div class="k-cap">Uncontrollables</div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- OTP Gauges ----------------
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <span style="color: #003865; font-size: 16px; font-weight: 600; font-family: 'DM Sans', sans-serif;">
            OTP Performance Indicators
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1: 
        adjusted_otp = max(gross_otp, net_otp) if pd.notna(gross_otp) and pd.notna(net_otp) else (net_otp if pd.notna(net_otp) else gross_otp)
        st.plotly_chart(make_semi_gauge("Adjusted OTP", adjusted_otp),
                       use_container_width=True, config={"displayModeBar": False}, key=f"{tab_name}_gauge_adjusted")
    with c2: 
        st.plotly_chart(make_semi_gauge("Controllable OTP", net_otp),
                       use_container_width=True, config={"displayModeBar": False}, key=f"{tab_name}_gauge_net")
    with c3: 
        st.plotly_chart(make_semi_gauge("Raw OTP", gross_otp),
                       use_container_width=True, config={"displayModeBar": False}, key=f"{tab_name}_gauge_gross")
    
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Methodology note
    with st.expander("📋 Methodology & Definitions"):
        st.markdown(f"""
        <div style="padding: 8px 0;">
            <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                <tr style="border-bottom: 1px solid #E5E7EB;">
                    <td style="padding: 10px 0; font-weight: 600; color: #003865; width: 30%;">Data Scope</td>
                    <td style="padding: 10px 0; color: #374151;">EMEA countries, STATUS = 440-BILLED</td>
                </tr>
                <tr style="border-bottom: 1px solid #E5E7EB;">
                    <td style="padding: 10px 0; font-weight: 600; color: #003865;">Time Grouping</td>
                    <td style="padding: 10px 0; color: #374151;">POD DATE/TIME → YYYY-MM</td>
                </tr>
                <tr style="border-bottom: 1px solid #E5E7EB;">
                    <td style="padding: 10px 0; font-weight: 600; color: #003865;">Raw OTP</td>
                    <td style="padding: 10px 0; color: #374151;">POD ≤ Target (UPD DEL or QDT)</td>
                </tr>
                <tr style="border-bottom: 1px solid #E5E7EB;">
                    <td style="padding: 10px 0; font-weight: 600; color: #003865;">Controllable OTP</td>
                    <td style="padding: 10px 0; color: #374151;">Excludes non-controllable delays (Agent, Customs, Warehouse)</td>
                </tr>
                <tr>
                    <td style="padding: 10px 0; font-weight: 600; color: #003865;">Volume</td>
                    <td style="padding: 10px 0; color: #374151;">Count of rows with valid POD date</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    
    # ---------------- Gross Metrics Section ----------------
    st.markdown(f"""
    <div style="margin: 36px 0 20px 0;">
        <span style="color: #003865; font-size: 18px; font-weight: 600; font-family: 'DM Sans', sans-serif;">
            Gross Metrics Overview
        </span>
        <span style="color: #6B7280; font-size: 13px; margin-left: 12px;">EMEA · All STATUS values</span>
    </div>
    """, unsafe_allow_html=True)
    
    if gross_df is not None and not gross_df.empty:
        gross_processed = preprocess(gross_df)
        
        if not gross_processed.empty:
            gross_vol_pod, gross_pieces_pod, _, _ = monthly_frames(gross_processed)
            
            col1, col2 = st.columns(2)
            
            # Gross Volume Chart - MBB Style
            with col1:
                if not gross_vol_pod.empty:
                    fig_gross_vol = go.Figure()
                    
                    fig_gross_vol.add_trace(go.Bar(
                        x=gross_vol_pod['Month_Display'],
                        y=gross_vol_pod['Volume'],
                        name='Volume',
                        marker_color=MARKEN_NAVY,
                        text=[f"{int(v):,}" for v in gross_vol_pod['Volume']],
                        textposition='outside',
                        textfont=dict(size=11, color=MARKEN_NAVY, family="DM Sans, sans-serif"),
                        hovertemplate="<b>%{x}</b><br>Volume: %{y:,}<extra></extra>"
                    ))
                    
                    layout = get_mbb_chart_layout("Gross Volume by Month", height=380, show_legend=False)
                    layout['yaxis']['title'] = "Volume"
                    fig_gross_vol.update_layout(**layout)
                    
                    st.plotly_chart(fig_gross_vol, use_container_width=True, key=f"{tab_name}_gross_vol")
            
            # Gross Pieces Chart - MBB Style
            with col2:
                if not gross_pieces_pod.empty:
                    fig_gross_pieces = go.Figure()
                    
                    fig_gross_pieces.add_trace(go.Bar(
                        x=gross_pieces_pod['Month_Display'],
                        y=gross_pieces_pod['Pieces'],
                        name='Pieces',
                        marker_color=MARKEN_GREEN,
                        text=[f"{int(v):,}" for v in gross_pieces_pod['Pieces']],
                        textposition='outside',
                        textfont=dict(size=11, color="#047857", family="DM Sans, sans-serif"),
                        hovertemplate="<b>%{x}</b><br>Pieces: %{y:,}<extra></extra>"
                    ))
                    
                    layout = get_mbb_chart_layout("Gross Pieces by Month", height=380, show_legend=False)
                    layout['yaxis']['title'] = "Pieces"
                    fig_gross_pieces.update_layout(**layout)
                    
                    st.plotly_chart(fig_gross_pieces, use_container_width=True, key=f"{tab_name}_gross_pieces")
    
    st.markdown("<hr>", unsafe_allow_html=True)

    # ---------------- Performance Analysis Section ----------------
    st.markdown(f"""
    <div style="margin: 36px 0 20px 0;">
        <span style="color: #003865; font-size: 18px; font-weight: 600; font-family: 'DM Sans', sans-serif;">
            Month-over-Month Performance
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    if 'TOTAL CHARGES' in processed_df.columns and 'ACCT NM' in processed_df.columns:
        monthly_changes = analyze_monthly_changes(processed_df)
        
        if not monthly_changes.empty:
            unique_months = monthly_changes.sort_values('Month_Sort')['Month'].unique()
            
            selected_month = st.selectbox(
                "Select analysis period",
                options=unique_months[1:],
                index=len(unique_months[1:]) - 1 if len(unique_months) > 1 else 0,
                key=f"{tab_name}_month_select"
            )
            
            if selected_month:
                create_performance_tables(monthly_changes, selected_month, tab_name)
            
            # Revenue Trend Chart - MBB Style
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="margin: 36px 0 20px 0;">
                <span style="color: #003865; font-size: 18px; font-weight: 600; font-family: 'DM Sans', sans-serif;">
                    Revenue Trend Analysis
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            if not revenue_pod.empty:
                fig_total_trend = go.Figure()
                
                fig_total_trend.add_trace(go.Bar(
                    x=revenue_pod['Month_Display'],
                    y=revenue_pod['Revenue'],
                    name='Revenue',
                    marker_color=MARKEN_NAVY,
                    opacity=0.85,
                    hovertemplate="<b>%{x}</b><br>Revenue: %{y:$,.0f}<extra></extra>"
                ))
                
                fig_total_trend.add_trace(go.Scatter(
                    x=revenue_pod['Month_Display'],
                    y=revenue_pod['Revenue'],
                    mode='lines+markers',
                    name='Trend',
                    line=dict(color=MARKEN_GREEN, width=3),
                    marker=dict(size=8, color=MARKEN_GREEN),
                    hoverinfo='skip'
                ))
                
                # Add revenue labels
                for i, row in revenue_pod.iterrows():
                    fig_total_trend.add_annotation(
                        x=row['Month_Display'],
                        y=row['Revenue'],
                        text=_revenue_fmt(row['Revenue']),
                        showarrow=False,
                        yshift=20,
                        font=dict(size=11, color=MARKEN_NAVY, family="DM Sans, sans-serif")
                    )
                
                # MoM growth annotations
                revenue_pod_copy = revenue_pod.copy()
                revenue_pod_copy['MoM_Growth'] = revenue_pod_copy['Revenue'].pct_change() * 100
                
                for i in range(1, len(revenue_pod_copy)):
                    growth = revenue_pod_copy.iloc[i]['MoM_Growth']
                    if pd.notna(growth):
                        color = "#10B981" if growth > 0 else "#EF4444"
                        fig_total_trend.add_annotation(
                            x=revenue_pod_copy.iloc[i]['Month_Display'],
                            y=revenue_pod_copy.iloc[i]['Revenue'],
                            text=f"{growth:+.1f}%",
                            showarrow=False,
                            yshift=40,
                            font=dict(size=10, color=color, family="Source Sans Pro, sans-serif"),
                            bgcolor="white",
                            bordercolor=color,
                            borderwidth=1,
                            borderpad=3
                        )
                
                layout = get_mbb_chart_layout("Monthly Revenue Performance", height=420)
                layout['yaxis']['title'] = "Revenue ($)"
                layout['yaxis']['tickformat'] = "$,.0f"
                fig_total_trend.update_layout(**layout)
                
                st.plotly_chart(fig_total_trend, use_container_width=True, key=f"{tab_name}_rev_trend")
                
                # Summary metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    avg_monthly = revenue_pod['Revenue'].mean()
                    st.metric("Avg Monthly", _revenue_fmt(avg_monthly))
                with col2:
                    if len(revenue_pod) > 1:
                        latest_growth = revenue_pod_copy.iloc[-1]['MoM_Growth']
                        st.metric("Latest MoM", f"{latest_growth:+.1f}%" if pd.notna(latest_growth) else "—")
                with col3:
                    max_revenue = revenue_pod['Revenue'].max()
                    max_month = revenue_pod.loc[revenue_pod['Revenue'].idxmax(), 'Month_Display']
                    st.metric("Peak Month", f"{max_month}")
                with col4:
                    total_period_revenue = revenue_pod['Revenue'].sum()
                    st.metric("Period Total", _revenue_fmt(total_period_revenue))

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---------------- OTP by Volume Chart - MBB Style ----------------
    st.markdown(f"""
    <div style="margin: 36px 0 20px 0;">
        <span style="color: #003865; font-size: 18px; font-weight: 600; font-family: 'DM Sans', sans-serif;">
            OTP Performance by Volume
        </span>
        <span style="color: #6B7280; font-size: 13px; margin-left: 12px;">Controllable OTP · POD Month</span>
    </div>
    """, unsafe_allow_html=True)
    
    if not vol_pod.empty and not otp_pod.empty:
        mv = vol_pod.merge(otp_pod[["Month_YYYY_MM","Net_OTP"]],
                           on="Month_YYYY_MM", how="left").sort_values("Month_Sort")
        x = mv["Month_Display"].tolist()
        y_vol = mv["Volume"].astype(float).tolist()
        y_net = mv["Net_OTP"].astype(float).tolist()

        fig = go.Figure()
        
        # Volume bars
        fig.add_trace(go.Bar(
            x=x, y=y_vol, name="Volume", 
            marker_color=MARKEN_NAVY,
            opacity=0.9,
            text=[f"{int(v):,}" for v in y_vol],
            textposition="inside",
            textfont=dict(size=12, color="white", family="DM Sans, sans-serif"),
            insidetextanchor="middle",
            yaxis="y",
            hovertemplate="<b>%{x}</b><br>Volume: %{y:,}<extra></extra>"
        ))
        
        # OTP line
        fig.add_trace(go.Scatter(
            x=x, y=y_net, name="Net OTP",
            mode="lines+markers", 
            line=dict(color=MARKEN_GREEN, width=3),
            marker=dict(size=10, color=MARKEN_GREEN, line=dict(width=2, color='white')),
            yaxis="y2",
            hovertemplate="<b>%{x}</b><br>OTP: %{y:.1f}%<extra></extra>"
        ))
        
        # OTP labels
        for xi, yi in zip(x, y_net):
            if pd.notna(yi):
                fig.add_annotation(
                    x=xi, y=yi, xref="x", yref="y2",
                    text=f"{yi:.1f}%",
                    showarrow=False,
                    yshift=18,
                    font=dict(size=11, color=MARKEN_GREEN, family="DM Sans, sans-serif", weight="bold"),
                    bgcolor="white",
                    borderpad=3
                )
        
        # Target line
        fig.add_shape(
            type="line", x0=-0.5, x1=len(x)-0.5,
            y0=float(otp_target), y1=float(otp_target),
            xref="x", yref="y2", 
            line=dict(color="#EF4444", dash="dash", width=2)
        )
        
        fig.add_annotation(
            x=len(x)-0.5, y=float(otp_target),
            xref="x", yref="y2",
            text=f"Target: {otp_target}%",
            showarrow=False,
            xshift=-60,
            font=dict(size=11, color="#EF4444"),
            bgcolor="white"
        )
        
        layout = get_mbb_chart_layout("", height=480)
        layout['yaxis'] = dict(
            title="Volume",
            side="left",
            gridcolor="#F3F4F6",
            showgrid=True,
            tickfont=dict(size=11, color="#6B7280"),
            title_font=dict(size=12, color="#374151")
        )
        layout['yaxis2'] = dict(
            title="OTP (%)",
            overlaying="y",
            side="right",
            range=[0, 110],
            showgrid=False,
            tickfont=dict(size=11, color="#6B7280"),
            title_font=dict(size=12, color="#374151")
        )
        layout['barmode'] = "overlay"
        fig.update_layout(**layout)
        
        st.plotly_chart(fig, use_container_width=True, key=f"{tab_name}_net_by_vol")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---------------- OTP by Pieces Chart - MBB Style ----------------
    st.markdown(f"""
    <div style="margin: 36px 0 20px 0;">
        <span style="color: #003865; font-size: 18px; font-weight: 600; font-family: 'DM Sans', sans-serif;">
            OTP Performance by Pieces
        </span>
        <span style="color: #6B7280; font-size: 13px; margin-left: 12px;">Controllable OTP · POD Month</span>
    </div>
    """, unsafe_allow_html=True)
    
    if not pieces_pod.empty and not otp_pod.empty:
        mp = pieces_pod.merge(otp_pod[["Month_YYYY_MM","Net_OTP"]],
                              on="Month_YYYY_MM", how="left").sort_values("Month_Sort")
        x = mp["Month_Display"].tolist()
        y_pcs = mp["Pieces"].astype(float).tolist()
        y_net = mp["Net_OTP"].astype(float).tolist()

        figp = go.Figure()
        
        figp.add_trace(go.Bar(
            x=x, y=y_pcs, name="Pieces", 
            marker_color=MARKEN_NAVY,
            opacity=0.9,
            text=[_kfmt(v) for v in y_pcs],
            textposition="inside",
            textfont=dict(size=12, color="white", family="DM Sans, sans-serif"),
            insidetextanchor="middle",
            yaxis="y",
            hovertemplate="<b>%{x}</b><br>Pieces: %{y:,.0f}<extra></extra>"
        ))
        
        figp.add_trace(go.Scatter(
            x=x, y=y_net, name="Net OTP",
            mode="lines+markers",
            line=dict(color=MARKEN_GREEN, width=3),
            marker=dict(size=10, color=MARKEN_GREEN, line=dict(width=2, color='white')),
            yaxis="y2",
            hovertemplate="<b>%{x}</b><br>OTP: %{y:.1f}%<extra></extra>"
        ))
        
        for xi, yi in zip(x, y_net):
            if pd.notna(yi):
                figp.add_annotation(
                    x=xi, y=yi, xref="x", yref="y2",
                    text=f"{yi:.1f}%",
                    showarrow=False,
                    yshift=18,
                    font=dict(size=11, color=MARKEN_GREEN, family="DM Sans, sans-serif", weight="bold"),
                    bgcolor="white",
                    borderpad=3
                )
        
        figp.add_shape(
            type="line", x0=-0.5, x1=len(x)-0.5,
            y0=float(otp_target), y1=float(otp_target),
            xref="x", yref="y2",
            line=dict(color="#EF4444", dash="dash", width=2)
        )
        
        figp.add_annotation(
            x=len(x)-0.5, y=float(otp_target),
            xref="x", yref="y2",
            text=f"Target: {otp_target}%",
            showarrow=False,
            xshift=-60,
            font=dict(size=11, color="#EF4444"),
            bgcolor="white"
        )
        
        layout = get_mbb_chart_layout("", height=480)
        layout['yaxis'] = dict(
            title="Pieces",
            side="left",
            gridcolor="#F3F4F6",
            showgrid=True,
            tickfont=dict(size=11, color="#6B7280"),
            title_font=dict(size=12, color="#374151")
        )
        layout['yaxis2'] = dict(
            title="OTP (%)",
            overlaying="y",
            side="right",
            range=[0, 110],
            showgrid=False,
            tickfont=dict(size=11, color="#6B7280"),
            title_font=dict(size=12, color="#374151")
        )
        layout['barmode'] = "overlay"
        figp.update_layout(**layout)
        
        st.plotly_chart(figp, use_container_width=True, key=f"{tab_name}_net_by_pcs")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---------------- OTP Trend Chart - MBB Style ----------------
    st.markdown(f"""
    <div style="margin: 36px 0 20px 0;">
        <span style="color: #003865; font-size: 18px; font-weight: 600; font-family: 'DM Sans', sans-serif;">
            OTP Trend Comparison
        </span>
        <span style="color: #6B7280; font-size: 13px; margin-left: 12px;">Gross vs Controllable · POD Month</span>
    </div>
    """, unsafe_allow_html=True)
    
    if not otp_pod.empty:
        otp_sorted = otp_pod.sort_values("Month_Sort")
        x       = otp_sorted["Month_Display"].tolist()
        gross_y = otp_sorted["Gross_OTP"].astype(float).tolist()
        net_y   = otp_sorted["Net_OTP"].astype(float).tolist()

        fig2 = go.Figure()
        
        # Net OTP (controllable) - primary line
        fig2.add_trace(go.Scatter(
            x=x, y=net_y, 
            mode="lines+markers", 
            name="Controllable OTP",
            line=dict(color=MARKEN_GREEN, width=3),
            marker=dict(size=10, color=MARKEN_GREEN, line=dict(width=2, color='white')),
            hovertemplate="<b>%{x}</b><br>Controllable: %{y:.1f}%<extra></extra>"
        ))
        
        # Gross OTP - secondary line
        fig2.add_trace(go.Scatter(
            x=x, y=gross_y, 
            mode="lines+markers", 
            name="Raw OTP",
            line=dict(color=MARKEN_LIGHT_BLUE, width=2, dash='dot'),
            marker=dict(size=8, color=MARKEN_LIGHT_BLUE),
            hovertemplate="<b>%{x}</b><br>Raw: %{y:.1f}%<extra></extra>"
        ))
        
        # Labels for Net OTP
        for xi, yi in zip(x, net_y):
            if pd.notna(yi):
                fig2.add_annotation(
                    x=xi, y=yi, xref="x", yref="y",
                    text=f"{yi:.1f}%",
                    showarrow=False,
                    yshift=18,
                    font=dict(size=11, color=MARKEN_GREEN, family="DM Sans, sans-serif"),
                    bgcolor="rgba(255,255,255,0.9)"
                )
        
        # Target line
        fig2.add_shape(
            type="line", x0=-0.5, x1=len(x)-0.5,
            y0=float(otp_target), y1=float(otp_target),
            xref="x", yref="y",
            line=dict(color="#EF4444", dash="dash", width=2)
        )
        
        fig2.add_annotation(
            x=len(x)-0.5, y=float(otp_target),
            xref="x", yref="y",
            text=f"Target: {otp_target}%",
            showarrow=False,
            xshift=-60,
            font=dict(size=11, color="#EF4444"),
            bgcolor="white"
        )
        
        layout = get_mbb_chart_layout("", height=420)
        layout['yaxis'] = dict(
            title="OTP (%)",
            range=[min(min(gross_y), min(net_y)) - 5, 105],
            gridcolor="#F3F4F6",
            showgrid=True,
            tickfont=dict(size=11, color="#6B7280"),
            title_font=dict(size=12, color="#374151")
        )
        fig2.update_layout(**layout)
        
        st.plotly_chart(fig2, use_container_width=True, key=f"{tab_name}_otp_trend")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ---------------- Worst Accounts Section - MBB Style ----------------
    st.markdown(f"""
    <div style="margin: 36px 0 20px 0;">
        <span style="color: #003865; font-size: 18px; font-weight: 600; font-family: 'DM Sans', sans-serif;">
            Accounts Requiring Attention
        </span>
        <span style="color: #6B7280; font-size: 13px; margin-left: 12px;">Bottom 5 by Controllable OTP</span>
    </div>
    """, unsafe_allow_html=True)
    
    if 'ACCT NM' in processed_df.columns:
        base = processed_df.dropna(subset=['_pod', '_target']).copy()
        if not base.empty:
            base['Month_Year'] = base['_pod'].dt.to_period('M')
            unique_periods = sorted(base['Month_Year'].unique())
            
            if unique_periods:
                selected_period = st.selectbox(
                    "Select period",
                    options=unique_periods,
                    format_func=lambda x: x.strftime('%B %Y'),
                    index=len(unique_periods)-1,
                    key=f"{tab_name}_worst_month_select"
                )
                
                month_df = base[base['Month_Year'] == selected_period]
                
                if not month_df.empty:
                    month_df['ACCT NM'] = month_df['ACCT NM'].astype(str).str.strip()
                    month_df = month_df[month_df['ACCT NM'].ne('')]
                    
                    if not month_df.empty:
                        grp = (month_df.groupby('ACCT NM', as_index=False)
                                      .agg(Net_OTP=('On_Time_Net', 'mean'),
                                           Volume=('On_Time_Net', 'size')))
                        grp['Net_OTP'] = grp['Net_OTP'] * 100
                        
                        grp = grp[grp['Net_OTP'].notna() & (grp['Net_OTP'] > 0)]
                        
                        if not grp.empty:
                            worst = grp.nsmallest(5, 'Net_OTP').copy()
                            worst['Net_OTP'] = worst['Net_OTP'].round(2)
                            
                            figw = go.Figure()
                            
                            # Color bars based on OTP level
                            colors = ['#EF4444' if otp < 80 else '#F59E0B' if otp < 90 else '#10B981' 
                                     for otp in worst['Net_OTP']]
                            
                            figw.add_trace(go.Bar(
                                x=worst['Net_OTP'],
                                y=worst['ACCT NM'],
                                orientation='h',
                                marker_color=colors,
                                text=[f"{otp:.1f}% • {int(v)} shipments" for otp, v in zip(worst['Net_OTP'], worst['Volume'])],
                                textposition='outside',
                                textfont=dict(size=12, color="#374151", family="Source Sans Pro, sans-serif"),
                                hovertemplate="<b>%{y}</b><br>OTP: %{x:.1f}%<br>Volume: %{customdata}<extra></extra>",
                                customdata=worst['Volume']
                            ))
                            
                            figw.add_shape(
                                type="line",
                                x0=float(otp_target), x1=float(otp_target),
                                y0=-0.5, y1=len(worst)-0.5,
                                xref="x", yref="y",
                                line=dict(color="#EF4444", dash="dash", width=2)
                            )
                            
                            figw.add_annotation(
                                x=float(otp_target), y=-0.7,
                                xref="x", yref="y",
                                text=f"Target: {otp_target}%",
                                showarrow=False,
                                font=dict(size=11, color="#EF4444"),
                                bgcolor="white"
                            )
                            
                            layout = get_mbb_chart_layout(f"{selected_period.strftime('%B %Y')} — Accounts Below Target", height=350, show_legend=False)
                            layout['xaxis'] = dict(
                                title="Controllable OTP (%)",
                                range=[0, 105],
                                gridcolor="#F3F4F6",
                                showgrid=True,
                                tickfont=dict(size=11, color="#6B7280")
                            )
                            layout['yaxis'] = dict(
                                title="",
                                automargin=True,
                                tickfont=dict(size=12, color="#374151", family="Source Sans Pro, sans-serif")
                            )
                            layout['margin'] = dict(l=10, r=80, t=60, b=60)
                            figw.update_layout(**layout)
                            
                            st.plotly_chart(figw, use_container_width=True, key=f"{tab_name}_worst5_chart")

    # QC breakdown
    if "QC_NAME_CLEAN" in processed_df.columns or "QC NAME" in processed_df.columns:
        qc_src = processed_df.copy()
        if "QC_NAME_CLEAN" not in qc_src.columns and "QC NAME" in qc_src.columns:
            qc_src["QC_NAME_CLEAN"] = qc_src["QC NAME"].astype(str)
        qc_src["Control_Type"] = qc_src["QC_NAME_CLEAN"].str.contains(CTRL_REGEX, na=False).map({True:"Controllable", False:"Non-Controllable"})
        qc_tbl = (qc_src.groupby(["Control_Type","QC_NAME_CLEAN"], dropna=False)
                        .size().reset_index(name="Count")
                        .sort_values(["Control_Type","Count"], ascending=[True, False]))
        with st.expander("QC Classification Detail"):
            st.dataframe(qc_tbl, use_container_width=True)

# ---------------- Sidebar ----------------
with st.sidebar:
    logo_image_sidebar = load_logo()
    if logo_image_sidebar:
        st.image(logo_image_sidebar, width=180)
        st.markdown("<hr style='margin: 16px 0;'>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 16px 0;">
            <span style="color: #8DC63F; font-size: 18px; font-weight: bold;">⬢⬢⬢</span><br>
            <span style="color: #003865; font-size: 18px; font-weight: 700;">MARKEN</span><br>
            <span style="color: #6B7280; font-size: 10px;">a UPS Company</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<hr style='margin: 16px 0;'>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="color: #003865; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px;">
        Data Upload
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["xlsx"], label_visibility="collapsed")
    
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="color: #003865; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px;">
        Configuration
    </div>
    """, unsafe_allow_html=True)
    otp_target = st.number_input("OTP Target (%)", min_value=0, max_value=100, value=OTP_TARGET, step=1)
    
    debug_mode = st.checkbox("Debug Mode", value=False)
    
    st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="color: #003865; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px;">
        About
    </div>
    <div style="font-size: 12px; color: #6B7280; line-height: 1.6;">
        <b>Scope:</b> EMEA Region<br>
        <b>Sectors:</b> Healthcare, Non-Healthcare, RadioPharma<br>
        <b>Filters:</b> 440-BILLED Status
    </div>
    """, unsafe_allow_html=True)

# ---------------- Landing Page ----------------
if not uploaded_file:
    st.markdown("""
    <div style="max-width: 800px; margin: 40px auto; text-align: center;">
        <div style="background: linear-gradient(135deg, #003865 0%, #0056A0 100%); border-radius: 12px; padding: 48px; color: white;">
            <div style="font-size: 48px; margin-bottom: 16px;">📊</div>
            <h2 style="color: white; font-size: 28px; margin-bottom: 12px;">Welcome to the OTP Dashboard</h2>
            <p style="color: rgba(255,255,255,0.8); font-size: 16px; margin-bottom: 24px;">
                Upload your TMS report to begin analyzing On-Time Performance metrics across Healthcare and Logistics sectors.
            </p>
            <div style="background: rgba(255,255,255,0.1); border-radius: 8px; padding: 20px; text-align: left;">
                <p style="color: rgba(255,255,255,0.9); font-size: 14px; margin-bottom: 8px; font-weight: 600;">Required file format:</p>
                <p style="color: rgba(255,255,255,0.7); font-size: 13px; margin: 0;">
                    Excel (.xlsx) with sheets: Aviation SVC, MNX Charter, AMS, LDN, Americas International Desk, RadioPharma
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Instructions using Streamlit columns
    st.markdown("""
    <h3 style="color: #003865; font-size: 18px; margin: 32px 0 20px 0; text-align: center;">📋 Data Preparation Steps</h3>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: white; border: 1px solid #E5E7EB; border-radius: 8px; padding: 20px; margin-bottom: 16px;">
            <div style="color: #8DC63F; font-size: 24px; font-weight: 700; margin-bottom: 8px;">1</div>
            <div style="color: #003865; font-weight: 600; margin-bottom: 4px;">Export from TMS</div>
            <div style="color: #6B7280; font-size: 13px;">Navigate to Reports → Shipment Report AH VAR</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: white; border: 1px solid #E5E7EB; border-radius: 8px; padding: 20px;">
            <div style="color: #8DC63F; font-size: 24px; font-weight: 700; margin-bottom: 8px;">3</div>
            <div style="color: #003865; font-weight: 600; margin-bottom: 4px;">Download All Desks</div>
            <div style="color: #6B7280; font-size: 13px;">Export data for all regional desks</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; border: 1px solid #E5E7EB; border-radius: 8px; padding: 20px; margin-bottom: 16px;">
            <div style="color: #8DC63F; font-size: 24px; font-weight: 700; margin-bottom: 8px;">2</div>
            <div style="color: #003865; font-weight: 600; margin-bottom: 4px;">Select Date Range</div>
            <div style="color: #6B7280; font-size: 13px;">Filter by DEL DATE ACT for desired period</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: white; border: 1px solid #E5E7EB; border-radius: 8px; padding: 20px;">
            <div style="color: #8DC63F; font-size: 24px; font-weight: 700; margin-bottom: 8px;">4</div>
            <div style="color: #003865; font-weight: 600; margin-bottom: 4px;">Merge & Upload</div>
            <div style="color: #6B7280; font-size: 13px;">Combine sheets into single Excel file</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.stop()

# ---------------- Main Processing ----------------
with st.spinner("Processing data..."):
    healthcare_df, non_healthcare_df, stats, healthcare_gross_df, non_healthcare_gross_df, account_classification, radiopharma_df, radiopharma_gross_df = read_and_combine_sheets(uploaded_file)

# Processing statistics
with st.expander("📈 Data Processing Summary"):
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total Rows", f"{stats.get('total_rows_raw', 0):,}")
    with col2:
        st.metric("Sheets", f"{len(stats.get('sheets_read', []))}")
    with col3:
        st.metric("EMEA", f"{stats.get('emea_rows', 0):,}")
    with col4:
        st.metric("Billed", f"{stats.get('status_filtered', 0):,}")
    with col5:
        st.metric("Healthcare", f"{stats.get('healthcare_rows', 0):,}")
    with col6:
        st.metric("Non-HC", f"{stats.get('non_healthcare_rows', 0):,}")
    
    if 'by_sheet' in stats:
        st.markdown("#### Sheet Breakdown")
        sheet_df = pd.DataFrame(stats['by_sheet']).T
        if 'raw_rows' in sheet_df.columns:
            sheet_df = sheet_df[['raw_rows', 'initial', 'emea', 'final']]
            sheet_df.columns = ['Raw', 'Initial', 'EMEA', 'Final']
        st.dataframe(sheet_df, use_container_width=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏥 Healthcare", "✈️ Non-Healthcare", "☢️ RadioPharma", "📋 Classification"])

with tab1:
    if not healthcare_df.empty:
        st.markdown(f"""
        <div style="color: #6B7280; font-size: 13px; margin-bottom: 20px;">
            {len(healthcare_df):,} entries · EMEA · 440-BILLED
        </div>
        """, unsafe_allow_html=True)
    create_dashboard_view(healthcare_df, "Healthcare", otp_target, healthcare_gross_df, debug_mode)

with tab2:
    if not non_healthcare_df.empty:
        st.markdown(f"""
        <div style="color: #6B7280; font-size: 13px; margin-bottom: 20px;">
            {len(non_healthcare_df):,} entries · EMEA · 440-BILLED
        </div>
        """, unsafe_allow_html=True)
    create_dashboard_view(non_healthcare_df, "Non-Healthcare", otp_target, non_healthcare_gross_df, debug_mode)

with tab3:
    if not radiopharma_df.empty:
        st.markdown(f"""
        <div style="color: #6B7280; font-size: 13px; margin-bottom: 20px;">
            {len(radiopharma_df):,} entries · EMEA · 440-BILLED
        </div>
        """, unsafe_allow_html=True)
        create_dashboard_view(radiopharma_df, "RadioPharma", otp_target, radiopharma_gross_df, debug_mode)
    else:
        st.info("No RadioPharma data available. Ensure your Excel file contains a 'RadioPharma' sheet with EMEA countries.")

with tab4:
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <span style="color: #003865; font-size: 18px; font-weight: 600; font-family: 'DM Sans', sans-serif;">
            Account Classification Overview
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    if not account_classification.empty:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_accounts = len(account_classification)
        healthcare_accounts = len(account_classification[account_classification['Classification'] == 'Healthcare'])
        non_healthcare_accounts = len(account_classification[account_classification['Classification'] == 'Non-Healthcare'])
        radiopharma_accounts = len(account_classification[account_classification['Classification'].str.contains('RadioPharma')])
        not_used_accounts = len(account_classification[account_classification['Classification'].str.contains('Not Used')])
        
        with col1:
            st.metric("Total", f"{total_accounts:,}")
        with col2:
            st.metric("Healthcare", f"{healthcare_accounts:,}")
        with col3:
            st.metric("Non-Healthcare", f"{non_healthcare_accounts:,}")
        with col4:
            st.metric("RadioPharma", f"{radiopharma_accounts:,}")
        with col5:
            st.metric("Excluded", f"{not_used_accounts:,}")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        classification_filter = st.selectbox(
            "Filter by classification",
            options=["All", "Healthcare", "Non-Healthcare", "RadioPharma", "Not Used (Non-EMEA)"],
            index=0
        )
        
        if classification_filter == "All":
            filtered_df = account_classification
        else:
            filtered_df = account_classification[account_classification['Classification'].str.contains(classification_filter)]
        
        search_term = st.text_input("Search accounts", "")
        if search_term:
            filtered_df = filtered_df[filtered_df['Account Name'].str.contains(search_term, case=False, na=False)]
        
        st.dataframe(filtered_df, use_container_width=True, height=500)
        
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"account_classification_{classification_filter.lower().replace(' ', '_')}.csv",
            mime="text/csv",
        )
