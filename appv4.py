# Code 2============================================================================
# IMPORTS
# ============================================================================
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Healthcare vs Non-Healthcare Dashboard", 
    page_icon="📊",
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# Color Palette
NAVY = "#0b1f44"
GOLD = "#f0b429"
BLUE = "#1f77b4"
GREEN = "#10b981"
SLATE = "#334155"
GRID = "#e5e7eb"
RED = "#dc2626"
EMERALD = "#059669"
PURPLE = "#7c3aed"
ORANGE = "#ea580c"

# OTP Target
OTP_TARGET = 95

# EMEA Countries
EMEA_COUNTRIES = {
    'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 
    'LT', 'LU', 'MT', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE', 'GB', 'UK', 'NO', 'CH', 'IS',
    'AL', 'AD', 'AM', 'AZ', 'BA', 'BY', 'GE', 'XK', 'LI', 'MD', 'MC', 'ME', 'MK', 'RU', 'SM', 'RS', 
    'TR', 'UA', 'VA',
    'AE', 'BH', 'EG', 'IQ', 'IR', 'IL', 'JO', 'KW', 'LB', 'OM', 'PS', 'QA', 'SA', 'SY', 'YE',
    'DZ', 'AO', 'BJ', 'BW', 'BF', 'BI', 'CM', 'CV', 'CF', 'TD', 'KM', 'CG', 'CD', 'DJ', 'GQ', 'ER',
    'ET', 'GA', 'GM', 'GH', 'GN', 'GW', 'CI', 'KE', 'LS', 'LR', 'LY', 'MG', 'MW', 'ML', 'MR', 'MU',
    'MA', 'MZ', 'NA', 'NE', 'NG', 'RW', 'ST', 'SN', 'SC', 'SL', 'SO', 'ZA', 'SS', 'SD', 'SZ', 'TZ',
    'TG', 'TN', 'UG', 'ZM', 'ZW'
}

# Healthcare Keywords
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

# Non-Healthcare Keywords
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

# Controllable Regex Pattern
CTRL_REGEX = re.compile(r"\b(agent|del\s*agt|delivery\s*agent|customs|warehouse|w/house)\b", re.I)

# ============================================================================
# CUSTOM CSS STYLES
# ============================================================================
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
.mom-header {background: linear-gradient(135deg, #0b1f44 0%, #1e40af 100%);color:white;padding:12px;border-radius:8px;margin-bottom:16px;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _excel_to_dt(s: pd.Series) -> pd.Series:
    """Robust datetime: parse; if many NaT, try Excel serials."""
    out = pd.to_datetime(s, errors="coerce")
    if out.isna().mean() > 0.5:
        num = pd.to_numeric(s, errors="coerce")
        out2 = pd.to_datetime("1899-12-30") + pd.to_timedelta(num, unit="D")
        out = out.where(~out.isna(), out2)
    return out


def _get_target_series(df: pd.DataFrame) -> pd.Series | None:
    """Get target date series from dataframe."""
    if "UPD DEL" in df.columns and df["UPD DEL"].notna().any():
        return df["UPD DEL"]
    if "QDT" in df.columns:
        return df["QDT"]
    return None


def _kfmt(n: float) -> str:
    """Format number with K suffix for thousands."""
    if pd.isna(n): 
        return ""
    try: 
        n = float(n)
    except: 
        return ""
    return f"{n/1000:.1f}K" if n >= 1000 else f"{n:.0f}"


def _revenue_fmt(n: float) -> str:
    """Format revenue numbers with K or M suffix."""
    if pd.isna(n): 
        return ""
    try: 
        n = float(n)
    except: 
        return ""
    if n >= 1000000:
        return f"${n/1000000:.1f}M"
    elif n >= 1000:
        return f"${n/1000:.1f}K"
    else:
        return f"${n:.0f}"


def format_number(n: float, is_currency: bool = False) -> str:
    """Format numbers with thousands separators and optional currency."""
    if pd.isna(n):
        return "—"
    try:
        n = float(n)
    except:
        return "—"
    
    if is_currency:
        if abs(n) >= 1000000:
            return f"${n:,.2f}"[:10] + "M" if abs(n) >= 1000000 else f"${n:,.2f}"
        else:
            return f"${n:,.2f}"
    else:
        return f"{n:,.0f}"


def format_delta(current: float, previous: float, is_currency: bool = False) -> str:
    """Format delta with proper sign and percentage."""
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return "—"
    
    delta = current - previous
    pct = ((current / previous) - 1) * 100
    
    sign = "+" if delta >= 0 else ""
    if is_currency:
        return f"{sign}{format_number(delta, True)} ({sign}{pct:.1f}%)"
    else:
        return f"{sign}{format_number(delta, False)} ({sign}{pct:.1f}%)"


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

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

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
    fig.add_annotation(
        text=f"{v:.2f}%", x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(size=26, color=NAVY, family="Arial Black")
    )
    fig.add_annotation(
        text=title, x=0.5, y=1.18, xref="paper", yref="paper",
        showarrow=False, font=dict(size=14, color=SLATE)
    )
    fig.update_layout(margin=dict(l=10, r=10, t=36, b=0), height=180)
    return fig

# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================

@st.cache_data(show_spinner=False)
def read_and_combine_sheets(uploaded):
    """Read Excel and split into healthcare and non-healthcare dataframes."""
    try:
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
        
        for sheet_name in sheet_names:
            try:
                df_sheet = pd.read_excel(uploaded, sheet_name=sheet_name, engine='openpyxl')
                initial_rows = len(df_sheet)
                df_sheet['Source_Sheet'] = sheet_name
                
                required_cols = ['PU CTRY', 'STATUS', 'POD DATE/TIME']
                missing_cols = [col for col in required_cols if col not in df_sheet.columns]
                if missing_cols:
                    st.warning(f"Sheet {sheet_name} missing columns: {missing_cols}")
                
                if 'PU CTRY' in df_sheet.columns:
                    df_sheet['PU CTRY'] = df_sheet['PU CTRY'].astype(str).str.strip().str.upper()
                    df_sheet_emea = df_sheet[df_sheet['PU CTRY'].isin(EMEA_COUNTRIES)]
                else:
                    df_sheet_emea = df_sheet
                
                emea_rows = len(df_sheet_emea)
                
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
                
                if len(df_sheet_final) > 0:
                    all_data.append(df_sheet_final)
                    
            except Exception as e:
                st.warning(f"Could not read sheet {sheet_name}: {str(e)}")
                stats['by_sheet'][sheet_name] = {
                    'initial': 0,
                    'emea': 0,
                    'final': 0
                }
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
        else:
            combined_df = pd.DataFrame()
        
        stats['total_rows'] = sum(s['initial'] for s in stats['by_sheet'].values())
        stats['emea_rows'] = sum(s['emea'] for s in stats['by_sheet'].values())
        stats['status_filtered'] = len(combined_df)
        
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
        
        return healthcare_df, non_healthcare_df, stats
    
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), {}


@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess dataframe for analysis."""
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
    """Build Monthly OTP, Volume, Pieces, Revenue dataframes."""
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
        otp["Net_OTP"] = (otp["Net_On"] / otp["Net_Tot"] * 100).round(2)

    vol, pieces, otp, revenue = [x.sort_values("Month_Sort") for x in (vol, pieces, otp, revenue)]
    return vol, pieces, otp, revenue


def calc_summary(d: pd.DataFrame):
    """Calculate summary statistics."""
    if d.empty:
        return np.nan, np.nan, 0, 0, 0, 0, 0
    
    base_otp = d.dropna(subset=["_pod","_target"])
    gross = base_otp["On_Time_Gross"].mean()*100 if len(base_otp) else np.nan
    net = base_otp["On_Time_Net"].mean()*100 if len(base_otp) else np.nan
    if pd.notna(gross) and pd.notna(net) and net < gross:
        net = gross
    late_df = base_otp[base_otp["Late"]]
    exceptions = int(len(late_df))
    controllables = int(late_df["Is_Controllable"].sum())
    uncontrollables = exceptions - controllables
    volume_total = int(len(d.dropna(subset=["_pod"])))
    total_revenue = float(d["TOTAL CHARGES"].sum()) if "TOTAL CHARGES" in d.columns else 0
    return (round(gross,2) if pd.notna(gross) else np.nan,
            round(net,2) if pd.notna(net) else np.nan,
            volume_total, exceptions, controllables, uncontrollables, total_revenue)


def analyze_monthly_changes(df: pd.DataFrame, metric: str = 'TOTAL CHARGES'):
    """Analyze month-over-month changes for accounts."""
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

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def create_enhanced_performance_tables(monthly_changes: pd.DataFrame, month: str, sector: str):
    """Create enhanced performance tables with extra columns and better formatting."""
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
    
    st.markdown(f'<div class="mom-header"><h3>📊 {sector} Performance Analysis - {month}</h3></div>', 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 💰 Revenue Performance")
        
        top_revenue = month_data.nlargest(10, 'Revenue_Change')[
            ['Account', 'Revenue', 'Volume', 'Revenue_Change', 'Volume_Change']
        ].copy()
        
        if not top_revenue.empty:
            st.markdown("**🔝 Top 10 Revenue Increases**")
            
            display_df = pd.DataFrame()
            display_df['Account'] = top_revenue['Account'].str[:25]
            display_df['Current Rev'] = top_revenue['Revenue'].apply(lambda x: format_number(x, True))
            display_df['Volume'] = top_revenue['Volume'].apply(lambda x: format_number(x, False))
            display_df['ΔRevenue'] = top_revenue['Revenue_Change'].apply(
                lambda x: format_number(x, True) if pd.notna(x) else "—"
            )
            display_df['ΔVolume'] = top_revenue['Volume_Change'].apply(
                lambda x: ("+" if x >= 0 else "") + format_number(x, False) if pd.notna(x) else "—"
            )
            
            st.dataframe(display_df.head(10), use_container_width=True, hide_index=True)
            
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=top_revenue['Revenue_Change'].head(5),
                y=top_revenue['Account'].str[:20].head(5),
                orientation='h',
                marker_color=GREEN,
                text=top_revenue['Revenue_Change'].apply(lambda x: format_number(x, True)).head(5),
                textposition='outside'
            ))
            fig_bar.update_layout(height=150, margin=dict(l=0, r=0, t=0, b=0), 
                                showlegend=False, plot_bgcolor="white",
                                xaxis=dict(visible=False), yaxis=dict(automargin=True))
            st.plotly_chart(fig_bar, use_container_width=True)
        
        worst_revenue = month_data.nsmallest(10, 'Revenue_Change')[
            ['Account', 'Revenue', 'Volume', 'Revenue_Change', 'Volume_Change']
        ].copy()
        
        if not worst_revenue.empty:
            st.markdown("**📉 Top 10 Revenue Decreases**")
            
            display_df = pd.DataFrame()
            display_df['Account'] = worst_revenue['Account'].str[:25]
            display_df['Current Rev'] = worst_revenue['Revenue'].apply(lambda x: format_number(x, True))
            display_df['Volume'] = worst_revenue['Volume'].apply(lambda x: format_number(x, False))
            display_df['ΔRevenue'] = worst_revenue['Revenue_Change'].apply(
                lambda x: format_number(x, True) if pd.notna(x) else "—"
            )
            display_df['ΔVolume'] = worst_revenue['Volume_Change'].apply(
                lambda x: ("+" if x >= 0 else "") + format_number(x, False) if pd.notna(x) else "—"
            )
            
            st.dataframe(display_df.head(10), use_container_width=True, hide_index=True)
            
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=worst_revenue['Revenue_Change'].head(5),
                y=worst_revenue['Account'].str[:20].head(5),
                orientation='h',
                marker_color=RED,
                text=worst_revenue['Revenue_Change'].apply(lambda x: format_number(x, True)).head(5),
                textposition='outside'
            ))
            fig_bar.update_layout(height=150, margin=dict(l=0, r=0, t=0, b=0),
                                showlegend=False, plot_bgcolor="white",
                                xaxis=dict(visible=False), yaxis=dict(automargin=True))
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        st.markdown("### 📦 Volume Performance")
        
        volume_data = month_data.dropna(subset=['Volume_Change'])
        if not volume_data.empty:
            top_volume = volume_data.nlargest(10, 'Volume_Change')[
                ['Account', 'Volume', 'Revenue', 'Volume_Change', 'Revenue_Change']
            ].copy()
            
            st.markdown("**🔝 Top 10 Volume Increases**")
            
            display_df = pd.DataFrame()
            display_df['Account'] = top_volume['Account'].str[:25]
            display_df['Volume'] = top_volume['Volume'].apply(lambda x: format_number(x, False))
            display_df['Revenue'] = top_volume['Revenue'].apply(lambda x: format_number(x, True))
            display_df['ΔVolume'] = top_volume['Volume_Change'].apply(
                lambda x: ("+" if x >= 0 else "") + format_number(x, False) if pd.notna(x) else "—"
            )
            display_df['ΔRevenue'] = top_volume['Revenue_Change'].apply(
                lambda x: format_number(x, True) if pd.notna(x) else "—"
            )
            
            st.dataframe(display_df.head(10), use_container_width=True, hide_index=True)
            
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=top_volume['Volume_Change'].head(5),
                y=top_volume['Account'].str[:20].head(5),
                orientation='h',
                marker_color=EMERALD,
                text=top_volume['Volume_Change'].apply(lambda x: format_number(x, False)).head(5),
                textposition='outside'
            ))
            fig_bar.update_layout(height=150, margin=dict(l=0, r=0, t=0, b=0),
                                showlegend=False, plot_bgcolor="white",
                                xaxis=dict(visible=False), yaxis=dict(automargin=True))
            st.plotly_chart(fig_bar, use_container_width=True)
            
            worst_volume = volume_data.nsmallest(10, 'Volume_Change')[
                ['Account', 'Volume', 'Revenue', 'Volume_Change', 'Revenue_Change']
            ].copy()
            
            st.markdown("**📉 Top 10 Volume Decreases**")
            
            display_df = pd.DataFrame()
            display_df['Account'] = worst_volume['Account'].str[:25]
            display_df['Volume'] = worst_volume['Volume'].apply(lambda x: format_number(x, False))
            display_df['Revenue'] = worst_volume['Revenue'].apply(lambda x: format_number(x, True))
            display_df['ΔVolume'] = worst_volume['Volume_Change'].apply(
                lambda x: format_number(x, False) if pd.notna(x) else "—"
            )
            display_df['ΔRevenue'] = worst_volume['Revenue_Change'].apply(
                lambda x: format_number(x, True) if pd.notna(x) else "—"
            )
            
            st.dataframe(display_df.head(10), use_container_width=True, hide_index=True)
            
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=worst_volume['Volume_Change'].head(5),
                y=worst_volume['Account'].str[:20].head(5),
                orientation='h',
                marker_color=ORANGE,
                text=worst_volume['Volume_Change'].apply(lambda x: format_number(x, False)).head(5),
                textposition='outside'
            ))
            fig_bar.update_layout(height=150, margin=dict(l=0, r=0, t=0, b=0),
                                showlegend=False, plot_bgcolor="white",
                                xaxis=dict(visible=False), yaxis=dict(automargin=True))
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with col3:
        st.markdown("### 📋 Pieces Performance")
        
        pieces_data = month_data.dropna(subset=['Pieces_Change'])
        if not pieces_data.empty:
            top_pieces = pieces_data.nlargest(10, 'Pieces_Change')[
                ['Account', 'Pieces', 'Revenue', 'Pieces_Change', 'Revenue_Change']
            ].copy()
            
            st.markdown("**🔝 Top 10 Pieces Increases**")
            
            display_df = pd.DataFrame()
            display_df['Account'] = top_pieces['Account'].str[:25]
            display_df['Pieces'] = top_pieces['Pieces'].apply(lambda x: format_number(x, False))
            display_df['Revenue'] = top_pieces['Revenue'].apply(lambda x: format_number(x, True))
            display_df['ΔPieces'] = top_pieces['Pieces_Change'].apply(
                lambda x: ("+" if x >= 0 else "") + format_number(x, False) if pd.notna(x) else "—"
            )
            display_df['ΔRevenue'] = top_pieces['Revenue_Change'].apply(
                lambda x: format_number(x, True) if pd.notna(x) else "—"
            )
            
            st.dataframe(display_df.head(10), use_container_width=True, hide_index=True)
            
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=top_pieces['Pieces_Change'].head(5),
                y=top_pieces['Account'].str[:20].head(5),
                orientation='h',
                marker_color=PURPLE,
                text=top_pieces['Pieces_Change'].apply(lambda x: format_number(x, False)).head(5),
                textposition='outside'
            ))
            fig_bar.update_layout(height=150, margin=dict(l=0, r=0, t=0, b=0),
                                showlegend=False, plot_bgcolor="white",
                                xaxis=dict(visible=False), yaxis=dict(automargin=True))
            st.plotly_chart(fig_bar, use_container_width=True)
            
            worst_pieces = pieces_data.nsmallest(10, 'Pieces_Change')[
                ['Account', 'Pieces', 'Revenue', 'Pieces_Change', 'Revenue_Change']
            ].copy()
            
            st.markdown("**📉 Top 10 Pieces Decreases**")
            
            display_df = pd.DataFrame()
            display_df['Account'] = worst_pieces['Account'].str[:25]
            display_df['Pieces'] = worst_pieces['Pieces'].apply(lambda x: format_number(x, False))
            display_df['Revenue'] = worst_pieces['Revenue'].apply(lambda x: format_number(x, True))
            display_df['ΔPieces'] = worst_pieces['Pieces_Change'].apply(
                lambda x: format_number(x, False) if pd.notna(x) else "—"
            )
            display_df['ΔRevenue'] = worst_pieces['Revenue_Change'].apply(
                lambda x: format_number(x, True) if pd.notna(x) else "—"
            )
            
            st.dataframe(display_df.head(10), use_container_width=True, hide_index=True)
            
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=worst_pieces['Pieces_Change'].head(5),
                y=worst_pieces['Account'].str[:20].head(5),
                orientation='h',
                marker_color=SLATE,
                text=worst_pieces['Pieces_Change'].apply(lambda x: format_number(x, False)).head(5),
                textposition='outside'
            ))
            fig_bar.update_layout(height=150, margin=dict(l=0, r=0, t=0, b=0),
                                showlegend=False, plot_bgcolor="white",
                                xaxis=dict(visible=False), yaxis=dict(automargin=True))
            st.plotly_chart(fig_bar, use_container_width=True)


def create_revenue_charts(processed_df: pd.DataFrame, sector: str):
    """Create monthly revenue bar chart and top 10 accounts histogram."""
    if 'TOTAL CHARGES' not in processed_df.columns or processed_df.empty:
        return
    
    monthly_revenue = processed_df.groupby(['Month_Display', 'Month_Sort'])['TOTAL CHARGES'].sum().reset_index()
    monthly_revenue = monthly_revenue.sort_values('Month_Sort')
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown(f"### 📈 {sector}: Monthly Total Revenue")
        
        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Bar(
            x=monthly_revenue['Month_Display'],
            y=monthly_revenue['TOTAL CHARGES'],
            marker_color=NAVY,
            text=monthly_revenue['TOTAL CHARGES'].apply(lambda x: format_number(x, True)),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>'
        ))
        
        fig_monthly.update_layout(
            height=400,
            plot_bgcolor="white",
            xaxis=dict(title="", tickangle=-45),
            yaxis=dict(title="Revenue ($)", gridcolor=GRID, showgrid=True),
            margin=dict(l=0, r=0, t=20, b=60)
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    with chart_col2:
        st.markdown(f"### 📊 {sector}: Top 10 Accounts Revenue Share")
        
        top_accounts = processed_df.groupby('ACCT NM')['TOTAL CHARGES'].sum().nlargest(10).reset_index()
        total_revenue = processed_df['TOTAL CHARGES'].sum()
        top_10_revenue = top_accounts['TOTAL CHARGES'].sum()
        top_10_percentage = (top_10_revenue / total_revenue * 100) if total_revenue > 0 else 0
        
        fig_top10 = go.Figure()
        fig_top10.add_trace(go.Bar(
            x=top_accounts['ACCT NM'].str[:20],
            y=top_accounts['TOTAL CHARGES'],
            marker_color=GOLD,
            text=top_accounts['TOTAL CHARGES'].apply(lambda x: format_number(x, True)),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.0f}<extra></extra>'
        ))
        
        fig_top10.add_annotation(
            text=f"Top 10 accounts: {top_10_percentage:.1f}% of total revenue",
            xref="paper", yref="paper",
            x=0.5, y=1.1,
            showarrow=False,
            font=dict(size=14, color=NAVY, family="Arial Black"),
            bgcolor="rgba(240, 180, 41, 0.2)",
            bordercolor=GOLD,
            borderwidth=2,
            borderpad=8
        )
        
        fig_top10.update_layout(
            height=400,
            plot_bgcolor="white",
            xaxis=dict(title="", tickangle=-45),
            yaxis=dict(title="Revenue ($)", gridcolor=GRID, showgrid=True),
            margin=dict(l=0, r=0, t=60, b=60)
        )
        st.plotly_chart(fig_top10, use_container_width=True)


def create_dashboard_view(df: pd.DataFrame, tab_name: str, otp_target: float, debug_mode: bool = False):
    """Create complete dashboard view for a sector."""
    if df.empty:
        st.warning(f"No data available for {tab_name} sector.")
        return
    
    processed_df = preprocess(df)
    
    if processed_df.empty:
        st.warning(f"No valid data after preprocessing for {tab_name}.")
        return
    
    vol_pod, pieces_pod, otp_pod, revenue_pod = monthly_frames(processed_df)
    
    gross_otp, net_otp, total_vol, total_exc, ctrl_exc, unctrl_exc, total_rev = calc_summary(processed_df)
    
    st.markdown("### 📊 Key Performance Indicators")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    with kpi1:
        st.markdown(f'<div class="kpi"><div class="k-num">{gross_otp:.2f}%</div><div class="k-cap">Gross OTP</div></div>', 
                    unsafe_allow_html=True)
    with kpi2:
        st.markdown(f'<div class="kpi"><div class="k-num">{net_otp:.2f}%</div><div class="k-cap">Net OTP</div></div>', 
                    unsafe_allow_html=True)
    with kpi3:
        st.markdown(f'<div class="kpi"><div class="k-num">{total_vol:,}</div><div class="k-cap">Total Volume</div></div>', 
                    unsafe_allow_html=True)
    with kpi4:
        st.markdown(f'<div class="kpi"><div class="k-num">{total_exc:,}</div><div class="k-cap">Exceptions</div></div>', 
                    unsafe_allow_html=True)
    with kpi5:
        st.markdown(f'<div class="kpi"><div class="k-num">{_revenue_fmt(total_rev)}</div><div class="k-cap">Total Revenue</div></div>', 
                    unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(make_semi_gauge("Gross OTP", gross_otp), use_container_width=True)
    with col2:
        st.plotly_chart(make_semi_gauge("Net OTP", net_otp), use_container_width=True)
    
    st.markdown("---")
    
    create_revenue_charts(processed_df, tab_name)
    
    st.markdown("---")
    
    if not processed_df.empty and 'Month_Display' in processed_df.columns:
        available_months = sorted(processed_df['Month_Display'].dropna().unique())
        if len(available_months) > 0:
            st.markdown("### 📈 Month-over-Month Performance Analysis")
            selected_month = st.selectbox(
                f"Select month for detailed analysis ({tab_name}):",
                options=available_months,
                index=len(available_months)-1,
                key=f"{tab_name}_mom_month_select"
            )
            
            monthly_changes = analyze_monthly_changes(processed_df)
            if not monthly_changes.empty:
                create_enhanced_performance_tables(monthly_changes, selected_month, tab_name)
    
    st.markdown("---")
    
    st.subheader(f"{tab_name}: Controllable (Net) OTP by Volume — POD Month")
    if not vol_pod.empty and not otp_pod.empty:
        mv = vol_pod.merge(otp_pod[["Month_YYYY_MM","Net_OTP"]],
                           on="Month_YYYY_MM", how="left").sort_values("Month_Sort")
        x = mv["Month_Display"].tolist()
        y_vol = mv["Volume"].astype(float).tolist()
        y_net = mv["Net_OTP"].astype(float).tolist()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=x, y=y_vol, name="Volume (Rows)", 
            marker_color=NAVY,
            text=[_kfmt(v) for v in y_vol],
            textposition="inside",
            textfont=dict(size=14, color="white", family="Arial Black"),
            textangle=0,
            yaxis="y",
            insidetextanchor="start"
        ))
        
        fig.add_trace(go.Scatter(
            x=x, y=y_net, name="Net OTP",
            mode="lines+markers", 
            line=dict(color=GOLD, width=3),
            marker=dict(size=10, color=GOLD),
            yaxis="y2"
        ))
        
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
        
        fig.add_shape(
            type="line", x0=-0.5, x1=len(x)-0.5,
            y0=float(otp_target), y1=float(otp_target),
            xref="x", yref="y2", 
            line=dict(color="red", dash="dash", width=2)
        )
        
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

    st.subheader(f"{tab_name}: Controllable (Net) OTP by Pieces — POD Month")
    if not pieces_pod.empty and not otp_pod.empty:
        mp = pieces_pod.merge(otp_pod[["Month_YYYY_MM","Net_OTP"]],
                              on="Month_YYYY_MM", how="left").sort_values("Month_Sort")
        x = mp["Month_Display"].tolist()
        y_pcs = mp["Pieces"].astype(float).tolist()
        y_net = mp["Net_OTP"].astype(float).tolist()

        figp = go.Figure()
        figp.add_trace(go.Bar(
            x=x, y=y_pcs, name="Pieces", 
            marker_color=NAVY,
            text=[_kfmt(v) for v in y_pcs],
            textposition="inside",
            textfont=dict(size=14, color="white", family="Arial Black"),
            textangle=0,
            yaxis="y",
            insidetextanchor="start"
        ))
        
        figp.add_trace(go.Scatter(
            x=x, y=y_net, name="Net OTP",
            mode="lines+markers",
            line=dict(color=GOLD, width=3),
            marker=dict(size=10, color=GOLD),
            yaxis="y2"
        ))
        
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
        
        figp.add_shape(
            type="line", x0=-0.5, x1=len(x)-0.5,
            y0=float(otp_target), y1=float(otp_target),
            xref="x", yref="y2",
            line=dict(color="red", dash="dash", width=2)
        )
        
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

    st.subheader(f"{tab_name}: Monthly OTP Trend (Gross vs Net) — POD Month")
    if not otp_pod.empty:
        otp_sorted = otp_pod.sort_values("Month_Sort")
        x = otp_sorted["Month_Display"].tolist()
        gross_y = otp_sorted["Gross_OTP"].astype(float).tolist()
        net_y = otp_sorted["Net_OTP"].astype(float).tolist()

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x, y=gross_y, mode="lines+markers", name="Gross OTP",
                                  line=dict(color=BLUE, width=3), marker=dict(size=8)))
        fig2.add_trace(go.Scatter(x=x, y=net_y, mode="lines+markers", name="Net OTP",
                                  line=dict(color=GREEN, width=3), marker=dict(size=8)))
        
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
        
        fig2.add_shape(
            type="line", x0=-0.5, x1=len(x)-0.5,
            y0=float(otp_target), y1=float(otp_target),
            xref="x", yref="y",
            line=dict(color="red", dash="dash", width=2)
        )
        
        fig2.add_annotation(
            x=len(x)-0.5, y=float(otp_target),
            xref="x", yref="y",
            text=f"Target: {otp_target}%",
            showarrow=False,
            xshift=-50,
            font=dict(size=12, color="red"),
            bgcolor="white"
        )
        
        fig2.update_layout(
            height=460,
            hovermode="x unified",
            plot_bgcolor="white",
            margin=dict(l=40, r=40, t=40, b=80),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
            xaxis=dict(title="", tickangle=-30, tickmode="array", tickvals=x, ticktext=x, automargin=True),
            yaxis=dict(title="OTP (%)", range=[0, 120], gridcolor=GRID, showgrid=True)
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No monthly OTP trend available.")

    st.subheader(f"{tab_name}: 5 Worst Accounts by Net OTP")
    
    if 'ACCT NM' in processed_df.columns:
        base = processed_df.dropna(subset=['_pod', '_target']).copy()
        if not base.empty:
            base['Month_Year'] = base['_pod'].dt.to_period('M')
            unique_periods = sorted(base['Month_Year'].unique())
            
            if unique_periods:
                selected_period = st.selectbox(
                    f"Select month for worst accounts analysis ({tab_name}):",
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
                            figw.add_trace(go.Bar(
                                x=worst['Net_OTP'],
                                y=worst['ACCT NM'],
                                orientation='h',
                                marker_color=NAVY,
                                text=[f"{otp:.2f}%  •  Vol {int(v)}" for otp, v in zip(worst['Net_OTP'], worst['Volume'])],
                                textposition='outside',
                                hovertemplate="<b>%{y}</b><br>Net OTP: %{x:.2f}%<br>Volume: %{customdata} rows<extra></extra>",
                                customdata=worst['Volume']
                            ))
                            
                            figw.add_shape(
                                type="line",
                                x0=float(otp_target), x1=float(otp_target),
                                y0=-0.5, y1=len(worst)-0.5,
                                xref="x", yref="y",
                                line=dict(color="red", dash="dash", width=2)
                            )
                            figw.add_annotation(
                                x=float(otp_target), y=-0.6,
                                xref="x", yref="y",
                                text=f"Target: {otp_target}%",
                                showarrow=False,
                                font=dict(size=12, color="red"),
                                bgcolor="white"
                            )
                            
                            figw.update_layout(
                                title_text=f"{selected_period.strftime('%B %Y')} — Worst 5 by Net OTP (with Volume)",
                                height=380,
                                plot_bgcolor="white",
                                margin=dict(l=10, r=40, t=40, b=40),
                                xaxis=dict(title="Net OTP (%)", range=[0, 110], gridcolor=GRID, showgrid=True),
                                yaxis=dict(title="", automargin=True)
                            )
                            st.plotly_chart(figw, use_container_width=True)
                            
                            st.caption(f"Worst 5 accounts — {selected_period.strftime('%B %Y')} (Net OTP and Volume)")
                            st.dataframe(
                                worst[['ACCT NM', 'Net_OTP', 'Volume']].rename(
                                    columns={'ACCT NM':'Account', 'Net_OTP':'Net OTP (%)'}
                                ),
                                use_container_width=True
                            )
                        else:
                            st.info(f"No non-null, >0% Net OTP accounts for {selected_period.strftime('%B %Y')}.")
                    else:
                        st.info(f"No valid account names for {selected_period.strftime('%B %Y')}.")
                else:
                    st.info(f"No data available for {selected_period.strftime('%B %Y')}.")
            else:
                st.info("No data with both POD and target available.")
        else:
            st.info("No rows with both POD and target available for account-level OTP.")
    else:
        st.info("Column 'ACCT NM' not found; cannot compute worst accounts.")

    if "QC_NAME_CLEAN" in processed_df.columns or "QC NAME" in processed_df.columns:
        qc_src = processed_df.copy()
        if "QC_NAME_CLEAN" not in qc_src.columns and "QC NAME" in qc_src.columns:
            qc_src["QC_NAME_CLEAN"] = qc_src["QC NAME"].astype(str)
        qc_src["Control_Type"] = qc_src["QC_NAME_CLEAN"].str.contains(CTRL_REGEX, na=False).map({True:"Controllable", False:"Non-Controllable"})
        qc_tbl = (qc_src.groupby(["Control_Type","QC_NAME_CLEAN"], dropna=False)
                        .size().reset_index(name="Count")
                        .sort_values(["Control_Type","Count"], ascending=[True, False]))
        with st.expander(f"{tab_name}: QC NAME breakdown (controllable vs non-controllable)"):
            st.dataframe(qc_tbl, use_container_width=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

st.title("Healthcare & Non-Healthcare Dashboard")

with st.sidebar:
    st.markdown("### 📁 Data Upload")
    uploaded_file = st.file_uploader("Upload Excel (.xlsx) file", type=["xlsx"])
    
    st.markdown("### ⚙️ Settings")
    otp_target = st.number_input("OTP Target (%)", min_value=0, max_value=100, value=OTP_TARGET, step=1)
    
    debug_mode = st.checkbox("Show Debug Information", value=False)
    
    st.markdown("---")
    st.markdown("### 📊 About this Dashboard")
    st.markdown("""
    This dashboard analyzes On-Time Performance (OTP) for:
    - **Healthcare**: Medical, pharmaceutical, and life science companies
    - **Non-Healthcare**: Aviation, logistics, and other industries
    
    **Data Scope:**
    - EMEA countries only
    - STATUS = 440-BILLED
    - Month grouping by POD DATE/TIME
    
    **Enhanced Features:**
    - Professional MoM analysis with formatted numbers
    - Revenue/Volume/Pieces performance tables
    - Monthly revenue trends and top account analysis
    - Compact bar visualizations
    """)

if not uploaded_file:
    st.info("""
    👆 **Please upload your Excel file to begin.**
    
    **Expected file structure:**
    - Multiple sheets (Aviation SVC, MNX Charter, AMS, LDN, Americas International Desk)
    - Required columns: PU CTRY, STATUS, POD DATE/TIME, ACCT NM
    - Optional columns: UPD DEL, QDT, QC NAME, PIECES, TOTAL CHARGES
    
    **Data processing:**
    - Filters for EMEA countries only
    - Filters for STATUS = 440-BILLED
    - Categorizes accounts as Healthcare or Non-Healthcare
    - Calculates OTP metrics by POD month
    - Enhanced month-over-month performance analysis with professional formatting
    """)
    st.stop()

with st.spinner("Processing Excel file..."):
    healthcare_df, non_healthcare_df, stats = read_and_combine_sheets(uploaded_file)

if debug_mode:
    with st.expander("🔍 Debug: Complete Data Flow"):
        st.write("**1. Sheets Read:**")
        for sheet_name in ['Aviation SVC', 'MNX Charter', 'AMS', 'LDN', 'Americas International Desk']:
            if sheet_name in stats.get('by_sheet', {}):
                sheet_stats = stats['by_sheet'][sheet_name]
                st.write(f"   {sheet_name}:")
                st.write(f"      - Initial: {sheet_stats['initial']:,} rows")
                st.write(f"      - After EMEA filter: {sheet_stats['emea']:,} rows")
                st.write(f"      - After 440-BILLED filter: {sheet_stats['final']:,} rows")
        
        st.write("\n**2. Combined Data:**")
        st.write(f"   Total combined (EMEA + 440-BILLED): {stats.get('status_filtered', 0):,} rows")
        
        st.write("\n**3. Healthcare Classification:**")
        st.write(f"   Healthcare: {stats.get('healthcare_rows', 0):,} rows")
        st.write(f"   Non-Healthcare: {stats.get('non_healthcare_rows', 0):,} rows")
        
        if not healthcare_df.empty and 'Source_Sheet' in healthcare_df.columns:
            st.write("\n**Healthcare by Source:**")
            hc_sources = healthcare_df['Source_Sheet'].value_counts()
            for sheet, count in hc_sources.items():
                st.write(f"   {sheet}: {count:,} rows")
        
        if not non_healthcare_df.empty and 'Source_Sheet' in non_healthcare_df.columns:
            st.write("\n**Non-Healthcare by Source:**")
            non_hc_sources = non_healthcare_df['Source_Sheet'].value_counts()
            for sheet, count in non_hc_sources.items():
                st.write(f"   {sheet}: {count:,} rows")

with st.expander("📈 Data Processing Statistics"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{stats.get('total_rows', 0):,}")
    with col2:
        st.metric("EMEA Rows", f"{stats.get('emea_rows', 0):,}")
    with col3:
        st.metric("440-BILLED", f"{stats.get('status_filtered', 0):,}")
    with col4:
        st.metric("HC / Non-HC", f"{stats.get('healthcare_rows', 0):,} / {stats.get('non_healthcare_rows', 0):,}")
    
    if 'by_sheet' in stats:
        st.markdown("#### Breakdown by Sheet:")
        sheet_df = pd.DataFrame(stats['by_sheet']).T
        sheet_df.columns = ['Initial Rows', 'After EMEA Filter', 'After Status Filter']
        st.dataframe(sheet_df)
        
        st.markdown("#### Validation:")
        total_processed = stats.get('healthcare_rows', 0) + stats.get('non_healthcare_rows', 0)
        st.write(f"✓ Total after filters: {stats.get('status_filtered', 0)}")
        st.write(f"✓ HC + Non-HC total: {total_processed}")
        if total_processed == stats.get('status_filtered', 0):
            st.success("✅ All filtered rows are categorized correctly!")
        else:
            st.warning(f"⚠️ Mismatch: {stats.get('status_filtered', 0) - total_processed} rows not categorized")

tab1, tab2 = st.tabs(["🏥 Healthcare", "✈️ Non-Healthcare"])

with tab1:
    st.markdown("## Healthcare Sector Analysis")
    if not healthcare_df.empty:
        st.markdown(f"**Total Healthcare Entries:** {len(healthcare_df):,}")
        with st.expander("Sample Healthcare Accounts"):
            if 'ACCT NM' in healthcare_df.columns:
                unique_accounts = healthcare_df['ACCT NM'].dropna().unique()[:20]
                st.write(", ".join(unique_accounts))
        
        if debug_mode:
            with st.expander("🔍 Debug: Healthcare POD Date Processing"):
                if 'POD DATE/TIME' in healthcare_df.columns:
                    sample_pod = healthcare_df[['POD DATE/TIME']].dropna().head(10)
                    st.write("Sample POD DATE/TIME values:")
                    st.dataframe(sample_pod)
                    
                    test_dates = _excel_to_dt(healthcare_df['POD DATE/TIME'].head(10))
                    st.write("Parsed dates:")
                    st.write(test_dates.to_list())
                    
                    st.write("Month grouping (YYYY-MM):")
                    st.write(test_dates.dt.to_period("M").astype(str).to_list())
    
    create_dashboard_view(healthcare_df, "Healthcare", otp_target, debug_mode)

with tab2:
    st.markdown("## Non-Healthcare Sector Analysis")
    if not non_healthcare_df.empty:
        st.markdown(f"**Total Non-Healthcare Entries:** {len(non_healthcare_df):,}")
        with st.expander("Sample Non-Healthcare Accounts"):
            if 'ACCT NM' in non_healthcare_df.columns:
                unique_accounts = non_healthcare_df['ACCT NM'].dropna().unique()[:20]
                st.write(", ".join(unique_accounts))
        
        if debug_mode:
            with st.expander("🔍 Debug: Non-Healthcare POD Date Processing"):
                if 'POD DATE/TIME' in non_healthcare_df.columns:
                    sample_pod = non_healthcare_df[['POD DATE/TIME']].dropna().head(10)
                    st.write("Sample POD DATE/TIME values:")
                    st.dataframe(sample_pod)
                    
                    test_dates = _excel_to_dt(non_healthcare_df['POD DATE/TIME'].head(10))
                    st.write("Parsed dates:")
                    st.write(test_dates.to_list())
                    
                    st.write("Month grouping (YYYY-MM):")
                    st.write(test_dates.dt.to_period("M").astype(str).to_list())
    
    create_dashboard_view(non_healthcare_df, "Non-Healthcare", otp_target, debug_mode)
