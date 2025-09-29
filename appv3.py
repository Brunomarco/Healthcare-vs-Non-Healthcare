import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
        return f"${n/1000000:.2f}M"
    elif n >= 1000:
        return f"${n/1000:.2f}K"
    else:
        return f"${n:.2f}"

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
    """Create professional executive-level performance tables with enhanced metrics"""
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
    
    # Executive Summary Header
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 10px; margin-bottom: 30px;">
        <h2 style="color: white; margin: 0;">📊 {sector} Performance Analysis</h2>
        <h3 style="color: #f0f0f0; margin: 5px 0 0 0; font-weight: 400;">Month-over-Month Comparison: {month}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different metrics
    tab1, tab2, tab3, tab4 = st.tabs(["💰 Revenue", "📦 Volume", "📋 Pieces", "📈 Visual Analysis"])
    
    # Revenue Performance Tab with Enhanced Tables
    with tab1:
        col1, col2 = st.columns(2)
        
        # Top Revenue Performers with Volume Data
        with col1:
            top_revenue = month_data.nlargest(10, 'Revenue_Change')[
                ['Account', 'Revenue', 'Revenue_Prev', 'Revenue_Change', 'Revenue_Change_Pct',
                 'Volume', 'Volume_Change', 'Volume_Change_Pct']
            ].copy()
            
            if not top_revenue.empty:
                st.markdown("""
                <div style="background: #f0fdf4; padding: 15px; border-radius: 8px; border-left: 4px solid #10b981;">
                    <h4 style="color: #059669; margin: 0;">🚀 Top 10 Revenue Gainers</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Enhanced table with proper formatting
                display_df = pd.DataFrame()
                display_df['Account'] = top_revenue['Account']
                display_df['Revenue'] = top_revenue['Revenue'].apply(lambda x: f"${x:,.2f}")
                display_df['Rev Change'] = top_revenue.apply(
                    lambda x: f"+${x['Revenue_Change']:,.2f} ({x['Revenue_Change_Pct']:.1f}%)" 
                    if x['Revenue_Change_Pct'] > 0 else f"${x['Revenue_Change']:,.2f} ({x['Revenue_Change_Pct']:.1f}%)", 
                    axis=1
                )
                display_df['Volume'] = top_revenue['Volume'].apply(lambda x: f"{int(x):,}")
                display_df['Vol Change'] = top_revenue.apply(
                    lambda x: f"+{int(x['Volume_Change']):,} ({x['Volume_Change_Pct']:.1f}%)" 
                    if pd.notna(x['Volume_Change_Pct']) and x['Volume_Change_Pct'] > 0 
                    else f"{int(x['Volume_Change']):,} ({x['Volume_Change_Pct']:.1f}%)" 
                    if pd.notna(x['Volume_Change_Pct']) else "N/A", 
                    axis=1
                )
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Worst Revenue Performers with Volume Data
        with col2:
            worst_revenue = month_data.nsmallest(10, 'Revenue_Change')[
                ['Account', 'Revenue', 'Revenue_Prev', 'Revenue_Change', 'Revenue_Change_Pct',
                 'Volume', 'Volume_Change', 'Volume_Change_Pct']
            ].copy()
            
            if not worst_revenue.empty:
                st.markdown("""
                <div style="background: #fef2f2; padding: 15px; border-radius: 8px; border-left: 4px solid #dc2626;">
                    <h4 style="color: #dc2626; margin: 0;">📉 Top 10 Revenue Declines</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Enhanced table with proper formatting
                display_df = pd.DataFrame()
                display_df['Account'] = worst_revenue['Account']
                display_df['Revenue'] = worst_revenue['Revenue'].apply(lambda x: f"${x:,.2f}")
                display_df['Rev Change'] = worst_revenue.apply(
                    lambda x: f"-${abs(x['Revenue_Change']):,.2f} ({x['Revenue_Change_Pct']:.1f}%)", 
                    axis=1
                )
                display_df['Volume'] = worst_revenue['Volume'].apply(lambda x: f"{int(x):,}")
                display_df['Vol Change'] = worst_revenue.apply(
                    lambda x: f"{int(x['Volume_Change']):,} ({x['Volume_Change_Pct']:.1f}%)" 
                    if pd.notna(x['Volume_Change_Pct']) else "N/A", 
                    axis=1
                )
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Volume Performance Tab with Revenue Data
    with tab2:
        volume_data = month_data.dropna(subset=['Volume_Change'])
        if not volume_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                top_volume = volume_data.nlargest(10, 'Volume_Change')[
                    ['Account', 'Volume', 'Volume_Prev', 'Volume_Change', 'Volume_Change_Pct',
                     'Revenue', 'Revenue_Change', 'Revenue_Change_Pct']
                ].copy()
                
                st.markdown("""
                <div style="background: #f0fdf4; padding: 15px; border-radius: 8px; border-left: 4px solid #10b981;">
                    <h4 style="color: #059669; margin: 0;">🚀 Top 10 Volume Gainers</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Enhanced table
                display_df = pd.DataFrame()
                display_df['Account'] = top_volume['Account']
                display_df['Volume'] = top_volume['Volume'].apply(lambda x: f"{int(x):,}")
                display_df['Vol Change'] = top_volume.apply(
                    lambda x: f"+{int(x['Volume_Change']):,} ({x['Volume_Change_Pct']:.1f}%)", 
                    axis=1
                )
                display_df['Revenue'] = top_volume['Revenue'].apply(lambda x: f"${x:,.2f}")
                display_df['Rev Change'] = top_volume.apply(
                    lambda x: f"${x['Revenue_Change']:,.2f} ({x['Revenue_Change_Pct']:.1f}%)" 
                    if pd.notna(x['Revenue_Change_Pct']) else "N/A", 
                    axis=1
                )
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            with col2:
                worst_volume = volume_data.nsmallest(10, 'Volume_Change')[
                    ['Account', 'Volume', 'Volume_Prev', 'Volume_Change', 'Volume_Change_Pct',
                     'Revenue', 'Revenue_Change', 'Revenue_Change_Pct']
                ].copy()
                
                st.markdown("""
                <div style="background: #fef2f2; padding: 15px; border-radius: 8px; border-left: 4px solid #dc2626;">
                    <h4 style="color: #dc2626; margin: 0;">📉 Top 10 Volume Declines</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Enhanced table
                display_df = pd.DataFrame()
                display_df['Account'] = worst_volume['Account']
                display_df['Volume'] = worst_volume['Volume'].apply(lambda x: f"{int(x):,}")
                display_df['Vol Change'] = worst_volume.apply(
                    lambda x: f"{int(x['Volume_Change']):,} ({x['Volume_Change_Pct']:.1f}%)", 
                    axis=1
                )
                display_df['Revenue'] = worst_volume['Revenue'].apply(lambda x: f"${x:,.2f}")
                display_df['Rev Change'] = worst_volume.apply(
                    lambda x: f"${x['Revenue_Change']:,.2f} ({x['Revenue_Change_Pct']:.1f}%)" 
                    if pd.notna(x['Revenue_Change_Pct']) else "N/A", 
                    axis=1
                )
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Pieces Performance Tab with Revenue Data
    with tab3:
        pieces_data = month_data.dropna(subset=['Pieces_Change'])
        if not pieces_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                top_pieces = pieces_data.nlargest(10, 'Pieces_Change')[
                    ['Account', 'Pieces', 'Pieces_Prev', 'Pieces_Change', 'Pieces_Change_Pct',
                     'Revenue', 'Revenue_Change', 'Revenue_Change_Pct']
                ].copy()
                
                st.markdown("""
                <div style="background: #f0fdf4; padding: 15px; border-radius: 8px; border-left: 4px solid #10b981;">
                    <h4 style="color: #059669; margin: 0;">🚀 Top 10 Pieces Gainers</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Enhanced table
                display_df = pd.DataFrame()
                display_df['Account'] = top_pieces['Account']
                display_df['Pieces'] = top_pieces['Pieces'].apply(lambda x: f"{int(x):,}")
                display_df['Pcs Change'] = top_pieces.apply(
                    lambda x: f"+{int(x['Pieces_Change']):,} ({x['Pieces_Change_Pct']:.1f}%)", 
                    axis=1
                )
                display_df['Revenue'] = top_pieces['Revenue'].apply(lambda x: f"${x:,.2f}")
                display_df['Rev Change'] = top_pieces.apply(
                    lambda x: f"${x['Revenue_Change']:,.2f} ({x['Revenue_Change_Pct']:.1f}%)" 
                    if pd.notna(x['Revenue_Change_Pct']) else "N/A", 
                    axis=1
                )
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            with col2:
                worst_pieces = pieces_data.nsmallest(10, 'Pieces_Change')[
                    ['Account', 'Pieces', 'Pieces_Prev', 'Pieces_Change', 'Pieces_Change_Pct',
                     'Revenue', 'Revenue_Change', 'Revenue_Change_Pct']
                ].copy()
                
                st.markdown("""
                <div style="background: #fef2f2; padding: 15px; border-radius: 8px; border-left: 4px solid #dc2626;">
                    <h4 style="color: #dc2626; margin: 0;">📉 Top 10 Pieces Declines</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Enhanced table
                display_df = pd.DataFrame()
                display_df['Account'] = worst_pieces['Account']
                display_df['Pieces'] = worst_pieces['Pieces'].apply(lambda x: f"{int(x):,}")
                display_df['Pcs Change'] = worst_pieces.apply(
                    lambda x: f"{int(x['Pieces_Change']):,} ({x['Pieces_Change_Pct']:.1f}%)", 
                    axis=1
                )
                display_df['Revenue'] = worst_pieces['Revenue'].apply(lambda x: f"${x:,.2f}")
                display_df['Rev Change'] = worst_pieces.apply(
                    lambda x: f"${x['Revenue_Change']:,.2f} ({x['Revenue_Change_Pct']:.1f}%)" 
                    if pd.notna(x['Revenue_Change_Pct']) else "N/A", 
                    axis=1
                )
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Visual Analysis Tab
    with tab4:
        # Combined waterfall chart for top changes
        st.markdown("### 📊 Overall Impact Analysis")
        
        # Create a combined view of biggest movers
        top_5_gainers = month_data.nlargest(5, 'Revenue_Change')[['Account', 'Revenue_Change']]
        top_5_losers = month_data.nsmallest(5, 'Revenue_Change')[['Account', 'Revenue_Change']]
        combined = pd.concat([top_5_gainers, top_5_losers])
        
        # Waterfall chart
        fig_waterfall = go.Figure(go.Waterfall(
            name="Revenue", orientation="v",
            x=combined['Account'],
            y=combined['Revenue_Change'],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#10b981"}},
            decreasing={"marker": {"color": "#dc2626"}},
            text=[f"${x:,.2f}" for x in combined['Revenue_Change']],
            textposition="outside"
        ))
        
        fig_waterfall.update_layout(
            title="Top 5 Gainers vs Top 5 Losers - Revenue Impact",
            height=500,
            showlegend=False,
            plot_bgcolor='white',
            xaxis_tickangle=-45,
            yaxis=dict(tickformat='$,.0f')
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)

def create_all_months_visualizations(processed_df, vol_pod, pieces_pod, revenue_pod, otp_pod, tab_name):
    """Create comprehensive charts showing all months data together"""
    
    st.markdown("### 🎯 All Months Overview - Comprehensive Analysis")
    
    # Monthly Revenue Bar Chart and Histogram
    st.markdown("### 📊 Revenue Analysis - All Months")
    
    if not revenue_pod.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig_bar = go.Figure()
            rev_sorted = revenue_pod.sort_values('Month_Sort')
            fig_bar.add_trace(go.Bar(
                x=rev_sorted['Month_Display'],
                y=rev_sorted['Revenue'],
                text=[f"${v:,.0f}" for v in rev_sorted['Revenue']],
                textposition='outside',
                marker=dict(
                    color=rev_sorted['Revenue'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Revenue ($)")
                ),
                hovertemplate='<b>%{x}</b><br>Revenue: $%{y:,.2f}<extra></extra>'
            ))
            
            # Add average line
            avg_revenue = np.mean(rev_sorted['Revenue'])
            fig_bar.add_hline(y=avg_revenue, line_dash="dash", line_color="red",
                              annotation_text=f"Average: ${avg_revenue:,.0f}")
            
            fig_bar.update_layout(
                title=f"{tab_name} - Monthly Revenue Performance",
                xaxis_title="Month",
                yaxis_title="Total Revenue ($)",
                height=500,
                plot_bgcolor='white',
                yaxis=dict(tickformat='$,.0f', gridcolor='#e5e5e5'),
                xaxis=dict(tickangle=-45)
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Histogram
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=processed_df['TOTAL CHARGES'],
                nbinsx=30,
                marker_color='#10b981',
                name="Revenue Distribution"
            ))
            fig_hist.update_layout(
                title=f"{tab_name} - Revenue Distribution (All Orders)",
                xaxis_title="Revenue per Order ($)",
                yaxis_title="Frequency",
                height=500,
                plot_bgcolor='white',
                xaxis=dict(tickformat='$,.0f'),
                showlegend=False
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    
    # Combined Monthly Trends
    st.markdown("### 📈 Comprehensive Monthly Performance")
    
    # Continuing the complete script...

    if not vol_pod.empty and not pieces_pod.empty and not revenue_pod.empty:
        # Create subplot figure with all metrics
        fig_all = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Revenue Trend", "Volume Trend", "Pieces Trend"),
            vertical_spacing=0.08,
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        # Sort all data by month
        rev_sorted = revenue_pod.sort_values('Month_Sort')
        vol_sorted = vol_pod.sort_values('Month_Sort')
        pcs_sorted = pieces_pod.sort_values('Month_Sort')
        
        # Revenue subplot
        fig_all.add_trace(
            go.Scatter(
                x=rev_sorted['Month_Display'],
                y=rev_sorted['Revenue'],
                mode='lines+markers+text',
                name='Revenue',
                line=dict(color='#667eea', width=3),
                marker=dict(size=10),
                text=[f"${v:,.0f}" for v in rev_sorted['Revenue']],
                textposition='top center',
                fill='tozeroy',
                fillcolor='rgba(102, 126, 234, 0.2)'
            ),
            row=1, col=1
        )
        
        # Volume subplot
        fig_all.add_trace(
            go.Scatter(
                x=vol_sorted['Month_Display'],
                y=vol_sorted['Volume'],
                mode='lines+markers+text',
                name='Volume',
                line=dict(color='#10b981', width=3),
                marker=dict(size=10),
                text=[f"{int(v):,}" for v in vol_sorted['Volume']],
                textposition='top center',
                fill='tozeroy',
                fillcolor='rgba(16, 185, 129, 0.2)'
            ),
            row=2, col=1
        )
        
        # Pieces subplot
        fig_all.add_trace(
            go.Scatter(
                x=pcs_sorted['Month_Display'],
                y=pcs_sorted['Pieces'],
                mode='lines+markers+text',
                name='Pieces',
                line=dict(color='#f59e0b', width=3),
                marker=dict(size=10),
                text=[f"{int(p):,}" for p in pcs_sorted['Pieces']],
                textposition='top center',
                fill='tozeroy',
                fillcolor='rgba(245, 158, 11, 0.2)'
            ),
            row=3, col=1
        )
        
        fig_all.update_xaxes(tickangle=-45)
        fig_all.update_yaxes(gridcolor='#e5e5e5', row=1, col=1, tickformat='$,.0f')
        fig_all.update_yaxes(gridcolor='#e5e5e5', row=2, col=1, tickformat=',')
        fig_all.update_yaxes(gridcolor='#e5e5e5', row=3, col=1, tickformat=',')
        
        fig_all.update_layout(
            height=900,
            showlegend=False,
            plot_bgcolor='white',
            title=f"{tab_name} - Complete Monthly Progression"
        )
        
        st.plotly_chart(fig_all, use_container_width=True)

# ---------------- IO ----------------
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

# ---------------- Prep (ROW-LEVEL; each row = one entry) ----------------
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
    
    # TOTAL CHARGES numeric - CRITICAL FOR REVENUE ANALYSIS
    if "TOTAL CHARGES" in d.columns:
        d["TOTAL CHARGES"] = pd.to_numeric(d["TOTAL CHARGES"], errors="coerce").fillna(0)
    else:
        d["TOTAL CHARGES"] = 0
        st.warning("TOTAL CHARGES column not found - revenue analysis will be limited")

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
    
    # Data Validation Section
    if debug_mode:
        with st.expander(f"🔍 Data Validation - {tab_name}"):
            st.write("**Data Processing Check:**")
            
            # Check date parsing
            valid_pods = processed_df['_pod'].notna().sum()
            total_rows = len(processed_df)
            st.write(f"- Total rows: {total_rows:,}")
            st.write(f"- Rows with valid POD dates: {valid_pods:,} ({valid_pods/total_rows*100:.1f}%)")
            
            # Check month assignment
            month_summary = processed_df.groupby('Month_Display').agg({
                '_pod': 'count',
                'PIECES': 'sum',
                'TOTAL CHARGES': 'sum'
            }).rename(columns={'_pod': 'Volume'})
            
            st.write("\n**Monthly Data Summary:**")
            display_summary = pd.DataFrame()
            display_summary['Month'] = month_summary.index
            display_summary['Volume'] = month_summary['Volume'].apply(lambda x: f"{int(x):,}")
            display_summary['Pieces'] = month_summary['PIECES'].apply(lambda x: f"{int(x):,}")
            display_summary['Revenue'] = month_summary['TOTAL CHARGES'].apply(lambda x: f"${x:,.2f}")
            st.dataframe(display_summary, use_container_width=True, hide_index=True)
            
            # September specific check
            sep_months = [m for m in processed_df['Month_Display'].unique() if 'Sep' in m]
            if sep_months:
                for sep_month in sep_months:
                    sep_data = processed_df[processed_df['Month_Display'] == sep_month]
                    st.write(f"\n**{sep_month} Data Check:**")
                    st.write(f"- Volume: {len(sep_data):,} orders")
                    st.write(f"- Pieces: {sep_data['PIECES'].sum():,.0f}")
                    st.write(f"- Revenue: ${sep_data['TOTAL CHARGES'].sum():,.2f}")
    
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

    # ---------------- NEW: Month-over-Month Performance Analysis ----------------
    st.subheader(f"📈 {tab_name}: Month-over-Month Performance Analysis")
    
    if 'TOTAL CHARGES' in processed_df.columns and 'ACCT NM' in processed_df.columns:
        monthly_changes = analyze_monthly_changes(processed_df)
        
        if not monthly_changes.empty:
            # Get unique months sorted
            unique_months = monthly_changes.sort_values('Month_Sort')['Month'].unique()
            
            # Executive Summary Cards
            st.markdown("### 📊 Executive Performance Summary")
            
            # Calculate overall metrics for latest month
            latest_month_data = monthly_changes[monthly_changes['Month'] == unique_months[-1]] if len(unique_months) > 0 else pd.DataFrame()
            
            if not latest_month_data.empty:
                col1, col2, col3, col4 = st.columns(4)
                
                total_rev_change = latest_month_data['Revenue_Change'].sum()
                total_vol_change = latest_month_data['Volume_Change'].sum()
                total_pcs_change = latest_month_data['Pieces_Change'].sum()
                accounts_growing = len(latest_month_data[latest_month_data['Revenue_Change'] > 0])
                
                with col1:
                    st.markdown(f"""
                    <div style="background: {'#f0fdf4' if total_rev_change > 0 else '#fef2f2'}; 
                                padding: 20px; border-radius: 10px; text-align: center;">
                        <h3 style="margin: 0; color: {'#059669' if total_rev_change > 0 else '#dc2626'};">
                            {_revenue_fmt(abs(total_rev_change))}
                        </h3>
                        <p style="margin: 5px 0 0 0; color: #6b7280; font-size: 14px;">Total Revenue Change</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="background: {'#f0fdf4' if total_vol_change > 0 else '#fef2f2'}; 
                                padding: 20px; border-radius: 10px; text-align: center;">
                        <h3 style="margin: 0; color: {'#059669' if total_vol_change > 0 else '#dc2626'};">
                            {int(abs(total_vol_change)):,}
                        </h3>
                        <p style="margin: 5px 0 0 0; color: #6b7280; font-size: 14px;">Volume Change</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div style="background: {'#f0fdf4' if total_pcs_change > 0 else '#fef2f2'}; 
                                padding: 20px; border-radius: 10px; text-align: center;">
                        <h3 style="margin: 0; color: {'#059669' if total_pcs_change > 0 else '#dc2626'};">
                            {int(abs(total_pcs_change)):,}
                        </h3>
                        <p style="margin: 5px 0 0 0; color: #6b7280; font-size: 14px;">Pieces Change</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    growth_pct = (accounts_growing / len(latest_month_data)) * 100 if len(latest_month_data) > 0 else 0
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 20px; border-radius: 10px; text-align: center;">
                        <h3 style="margin: 0; color: white;">
                            {growth_pct:.1f}%
                        </h3>
                        <p style="margin: 5px 0 0 0; color: #f0f0f0; font-size: 14px;">Accounts Growing</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Add comprehensive all-months visualizations
            create_all_months_visualizations(processed_df, vol_pod, pieces_pod, revenue_pod, otp_pod, tab_name)
            
            st.markdown("---")
            
            # Create a selectbox for month selection
            selected_month = st.selectbox(
                f"Select month for detailed {tab_name} performance analysis:",
                options=unique_months[1:],  # Skip first month (no MoM data)
                index=len(unique_months[1:]) - 1 if len(unique_months) > 1 else 0,  # Default to latest
                key=f"{tab_name}_month_select"
            )
            
            if selected_month:
                create_performance_tables(monthly_changes, selected_month, tab_name)
    
    else:
        st.info(f"Revenue data (TOTAL CHARGES) not available for {tab_name} performance analysis")

    st.markdown("---")

    # [Continue with existing charts - Net OTP by Volume, Pieces, etc...]
    # [These remain the same as in your original code]

# ---------------- Main Application ----------------
# Sidebar
with st.sidebar:
    st.markdown("### 📁 Data Upload")
    uploaded_file = st.file_uploader("Upload Excel (.xlsx) file", type=["xlsx"])
    
    st.markdown("### ⚙️ Settings")
    otp_target = st.number_input("OTP Target (%)", min_value=0, max_value=100, value=OTP_TARGET, step=1)
    
    # Debug mode checkbox
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
    - Month-over-month revenue, volume, and pieces analysis
    - Top/worst performers with cross-metric visibility
    - Executive-level visualizations
    - Comprehensive data validation
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
    - Analyzes month-over-month performance changes
    """)
    st.stop()

# Process uploaded file
with st.spinner("Processing Excel file..."):
    healthcare_df, non_healthcare_df, stats = read_and_combine_sheets(uploaded_file)

# Show processing statistics
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

# Create tabs
tab1, tab2 = st.tabs(["🏥 Healthcare", "✈️ Non-Healthcare"])

with tab1:
    st.markdown("## Healthcare Sector Analysis")
    if not healthcare_df.empty:
        st.markdown(f"**Total Healthcare Entries:** {len(healthcare_df):,}")
        create_dashboard_view(healthcare_df, "Healthcare", otp_target, debug_mode)

with tab2:
    st.markdown("## Non-Healthcare Sector Analysis")
    if not non_healthcare_df.empty:
        st.markdown(f"**Total Non-Healthcare Entries:** {len(non_healthcare_df):,}")
        create_dashboard_view(non_healthcare_df, "Non-Healthcare", otp_target, debug_mode)
