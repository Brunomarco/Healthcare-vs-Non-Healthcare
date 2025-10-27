#Code 1 - Complete Version - ALL ROWS GUARANTEED
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
.dataframe {font-size: 14px !important;}
.dataframe td {padding: 8px !important;}
.dataframe th {padding: 10px !important; background-color: #f3f4f6 !important; font-weight: 600 !important;}
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
    
    # Revenue Performance (with Volume Change added)
    with col1:
        st.markdown("### 💰 Revenue Performance")
        
        # Top 10 Revenue Increases
        top_revenue = month_data.nlargest(10, 'Revenue_Change')[
            ['Account', 'Revenue', 'Revenue_Prev', 'Revenue_Change', 'Revenue_Change_Pct', 'Volume_Change', 'Volume_Change_Pct']
        ].copy()
        
        if not top_revenue.empty:
            st.markdown("**🔝 Top 10 Revenue Increases**")
            # Format for display
            top_revenue['Rev Change'] = top_revenue.apply(
                lambda x: f"+{_revenue_fmt(x['Revenue_Change'])} ({x['Revenue_Change_Pct']:.1f}%)" 
                if pd.notna(x['Revenue_Change_Pct']) else f"+{_revenue_fmt(x['Revenue_Change'])}", axis=1
            )
            top_revenue['Vol Change'] = top_revenue.apply(
                lambda x: f"{int(x['Volume_Change']):+d} ({x['Volume_Change_Pct']:.1f}%)" 
                if pd.notna(x['Volume_Change_Pct']) and pd.notna(x['Volume_Change']) 
                else "N/A", axis=1
            )
            top_revenue['Current'] = top_revenue['Revenue'].apply(_revenue_fmt)
            
            # Create styled dataframe with increased height
            st.dataframe(
                top_revenue[['Account', 'Current', 'Rev Change', 'Vol Change']].head(10),
                use_container_width=True,
                hide_index=True,
                height=400  # Increased height to show all data without scrolling
            )
        
        # Worst 10 Revenue Decreases
        worst_revenue = month_data.nsmallest(10, 'Revenue_Change')[
            ['Account', 'Revenue', 'Revenue_Prev', 'Revenue_Change', 'Revenue_Change_Pct', 'Volume_Change', 'Volume_Change_Pct']
        ].copy()
        
        if not worst_revenue.empty:
            st.markdown("**📉 Top 10 Revenue Decreases**")
            worst_revenue['Rev Change'] = worst_revenue.apply(
                lambda x: f"{_revenue_fmt(x['Revenue_Change'])} ({x['Revenue_Change_Pct']:.1f}%)" 
                if pd.notna(x['Revenue_Change_Pct']) else f"{_revenue_fmt(x['Revenue_Change'])}", axis=1
            )
            worst_revenue['Vol Change'] = worst_revenue.apply(
                lambda x: f"{int(x['Volume_Change']):+d} ({x['Volume_Change_Pct']:.1f}%)" 
                if pd.notna(x['Volume_Change_Pct']) and pd.notna(x['Volume_Change']) 
                else "N/A", axis=1
            )
            worst_revenue['Current'] = worst_revenue['Revenue'].apply(_revenue_fmt)
            
            st.dataframe(
                worst_revenue[['Account', 'Current', 'Rev Change', 'Vol Change']].head(10),
                use_container_width=True,
                hide_index=True,
                height=400  # Increased height to show all data without scrolling
            )
    
    # Volume Performance (with Revenue Change added)
    with col2:
        st.markdown("### 📦 Volume Performance")
        
        # Top 10 Volume Increases
        volume_data = month_data.dropna(subset=['Volume_Change'])
        if not volume_data.empty:
            top_volume = volume_data.nlargest(10, 'Volume_Change')[
                ['Account', 'Volume', 'Volume_Change', 'Volume_Change_Pct', 'Revenue_Change', 'Revenue_Change_Pct']
            ].copy()
            
            st.markdown("**🔝 Top 10 Volume Increases**")
            top_volume['Vol Change'] = top_volume.apply(
                lambda x: f"+{int(x['Volume_Change'])} ({x['Volume_Change_Pct']:.1f}%)" 
                if pd.notna(x['Volume_Change_Pct']) else f"+{int(x['Volume_Change'])}", axis=1
            )
            top_volume['Rev Change'] = top_volume.apply(
                lambda x: f"{_revenue_fmt(x['Revenue_Change'])} ({x['Revenue_Change_Pct']:.1f}%)" 
                if pd.notna(x['Revenue_Change_Pct']) and pd.notna(x['Revenue_Change'])
                else "N/A", axis=1
            )
            
            st.dataframe(
                top_volume[['Account', 'Volume', 'Vol Change', 'Rev Change']].head(10),
                use_container_width=True,
                hide_index=True,
                height=400  # Increased height to show all data without scrolling
            )
            
            # Worst 10 Volume Decreases
            worst_volume = volume_data.nsmallest(10, 'Volume_Change')[
                ['Account', 'Volume', 'Volume_Change', 'Volume_Change_Pct', 'Revenue_Change', 'Revenue_Change_Pct']
            ].copy()
            
            st.markdown("**📉 Top 10 Volume Decreases**")
            worst_volume['Vol Change'] = worst_volume.apply(
                lambda x: f"{int(x['Volume_Change'])} ({x['Volume_Change_Pct']:.1f}%)" 
                if pd.notna(x['Volume_Change_Pct']) else f"{int(x['Volume_Change'])}", axis=1
            )
            worst_volume['Rev Change'] = worst_volume.apply(
                lambda x: f"{_revenue_fmt(x['Revenue_Change'])} ({x['Revenue_Change_Pct']:.1f}%)" 
                if pd.notna(x['Revenue_Change_Pct']) and pd.notna(x['Revenue_Change'])
                else "N/A", axis=1
            )
            
            st.dataframe(
                worst_volume[['Account', 'Volume', 'Vol Change', 'Rev Change']].head(10),
                use_container_width=True,
                hide_index=True,
                height=400  # Increased height to show all data without scrolling
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
                hide_index=True,
                height=400  # Increased height to show all data without scrolling
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
                hide_index=True,
                height=400  # Increased height to show all data without scrolling
            )

# ---------------- IO - CRITICAL FUNCTION ----------------
@st.cache_data(show_spinner=False)
def read_and_combine_sheets(uploaded):
    """Read ALL rows from ALL sheets, then apply filters."""
    try:
        # Read ALL sheet names from the Excel file
        xl_file = pd.ExcelFile(uploaded, engine='openpyxl')
        all_sheet_names = xl_file.sheet_names
        
        all_data_raw = []  # Store ALL raw data first
        all_data_filtered = []  # Store filtered data
        
        stats = {
            'total_rows_raw': 0,  # ALL rows before any filtering
            'total_rows': 0,
            'emea_rows': 0,
            'status_filtered': 0,
            'healthcare_rows': 0,
            'non_healthcare_rows': 0,
            'by_sheet': {},
            'sheets_read': all_sheet_names
        }
        
        # Process EVERY sheet in the Excel file
        for sheet_name in all_sheet_names:
            try:
                # Read ALL rows from the sheet - NO FILTERING AT THIS STAGE
                df_sheet_raw = pd.read_excel(uploaded, sheet_name=sheet_name, engine='openpyxl')
                
                # Skip completely empty sheets
                if df_sheet_raw.empty or len(df_sheet_raw) == 0:
                    stats['by_sheet'][sheet_name] = {
                        'raw_rows': 0,
                        'initial': 0,
                        'emea': 0,
                        'final': 0,
                        'note': 'Empty sheet'
                    }
                    continue
                
                # Count ALL raw rows
                raw_row_count = len(df_sheet_raw)
                stats['total_rows_raw'] += raw_row_count
                
                # Add source sheet information to EVERY row
                df_sheet_raw['Source_Sheet'] = sheet_name
                
                # Store the raw data
                all_data_raw.append(df_sheet_raw.copy())
                
                # Now apply filters for the filtered version
                df_sheet = df_sheet_raw.copy()
                initial_rows = len(df_sheet)
                stats['total_rows'] += initial_rows
                
                # Apply EMEA filter ONLY if PU CTRY column exists
                if 'PU CTRY' in df_sheet.columns:
                    df_sheet['PU CTRY'] = df_sheet['PU CTRY'].astype(str).str.strip().str.upper()
                    # Replace nan/None with empty string to avoid filtering them out
                    df_sheet['PU CTRY'] = df_sheet['PU CTRY'].replace(['NAN', 'NONE', '<NA>'], '')
                    df_sheet_emea = df_sheet[
                        (df_sheet['PU CTRY'].isin(EMEA_COUNTRIES)) | 
                        (df_sheet['PU CTRY'] == '') |
                        (df_sheet['PU CTRY'].isna())
                    ]
                else:
                    # No PU CTRY column means keep ALL rows
                    df_sheet_emea = df_sheet
                
                emea_rows = len(df_sheet_emea)
                
                # Apply STATUS filter ONLY if STATUS column exists
                if 'STATUS' in df_sheet_emea.columns:
                    # Clean STATUS values
                    df_sheet_emea['STATUS'] = df_sheet_emea['STATUS'].astype(str).str.strip()
                    # Keep 440-BILLED and also rows with empty/missing status
                    df_sheet_final = df_sheet_emea[
                        (df_sheet_emea['STATUS'] == '440-BILLED') |
                        (df_sheet_emea['STATUS'] == '') |
                        (df_sheet_emea['STATUS'] == 'nan') |
                        (df_sheet_emea['STATUS'].isna())
                    ]
                else:
                    # No STATUS column means keep ALL rows
                    df_sheet_final = df_sheet_emea
                
                final_rows = len(df_sheet_final)
                
                stats['by_sheet'][sheet_name] = {
                    'raw_rows': raw_row_count,
                    'initial': initial_rows,
                    'emea': emea_rows,
                    'final': final_rows
                }
                
                # Add to combined filtered data
                if len(df_sheet_final) > 0:
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
        
        # Combine ALL raw data
        if all_data_raw:
            combined_raw_df = pd.concat(all_data_raw, ignore_index=True)
            stats['total_combined_raw'] = len(combined_raw_df)
        else:
            combined_raw_df = pd.DataFrame()
            stats['total_combined_raw'] = 0
        
        # Combine filtered data
        if all_data_filtered:
            combined_df = pd.concat(all_data_filtered, ignore_index=True)
        else:
            combined_df = pd.DataFrame()
        
        stats['emea_rows'] = sum(s.get('emea', 0) for s in stats['by_sheet'].values())
        stats['status_filtered'] = len(combined_df)
        
        # Show warning if significant data loss
        if stats['total_rows_raw'] > 0:
            retention_rate = (stats['status_filtered'] / stats['total_rows_raw']) * 100
            if retention_rate < 50:
                st.warning(f"⚠️ Only {retention_rate:.1f}% of raw data retained after filtering. Consider reviewing filter criteria.")
        
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
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
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
                           use_container_width=True, config={"displayModeBar": False}, key=f"{tab_name}_gauge_adjusted")
        with c2: 
            st.plotly_chart(make_semi_gauge("Controllable OTP", net_otp),
                           use_container_width=True, config={"displayModeBar": False}, key=f"{tab_name}_gauge_net")
        with c3: 
            st.plotly_chart(make_semi_gauge("Raw OTP", gross_otp),
                           use_container_width=True, config={"displayModeBar": False}, key=f"{tab_name}_gauge_gross" )

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
    st.markdown("---")
    st.subheader(f"📈 {tab_name}: Month-over-Month Performance Analysis")
    
    if 'TOTAL CHARGES' in processed_df.columns and 'ACCT NM' in processed_df.columns:
        monthly_changes = analyze_monthly_changes(processed_df)
        
        if not monthly_changes.empty:
            # Get unique months sorted
            unique_months = monthly_changes.sort_values('Month_Sort')['Month'].unique()
            
            # Create a selectbox for month selection
            selected_month = st.selectbox(
                f"Select month for {tab_name} performance analysis:",
                options=unique_months[1:],  # Skip first month (no MoM data)
                index=len(unique_months[1:]) - 1 if len(unique_months) > 1 else 0,  # Default to latest
                key=f"{tab_name}_month_select"
            )
            
            if selected_month:
                create_performance_tables(monthly_changes, selected_month, tab_name)
            
            # UPDATED: Total Revenue Trend by Month (not by account)
            st.markdown("---")
            st.subheader(f"📊 {tab_name}: Total Monthly Revenue Trend")
            
            # Group by month for total revenue
            if not revenue_pod.empty:
                # Create line chart for total revenue trend
                fig_total_trend = go.Figure()
                
                # Add bar chart for revenue
                fig_total_trend.add_trace(go.Bar(
                    x=revenue_pod['Month_Display'],
                    y=revenue_pod['Revenue'],
                    name='Monthly Revenue',
                    marker_color=NAVY,
                    text=[_revenue_fmt(v) for v in revenue_pod['Revenue']],
                    textposition='outside'
                ))
                
                # Add line for trend
                fig_total_trend.add_trace(go.Scatter(
                    x=revenue_pod['Month_Display'],
                    y=revenue_pod['Revenue'],
                    mode='lines+markers',
                    name='Trend',
                    line=dict(color=GOLD, width=3),
                    marker=dict(size=8, color=GOLD)
                ))
                
                # Calculate month-over-month growth rates
                revenue_pod_copy = revenue_pod.copy()
                revenue_pod_copy['MoM_Growth'] = revenue_pod_copy['Revenue'].pct_change() * 100
                
                # Add annotations for growth rates
                for i in range(1, len(revenue_pod_copy)):
                    growth = revenue_pod_copy.iloc[i]['MoM_Growth']
                    if pd.notna(growth):
                        color = GREEN if growth > 0 else RED
                        fig_total_trend.add_annotation(
                            x=revenue_pod_copy.iloc[i]['Month_Display'],
                            y=revenue_pod_copy.iloc[i]['Revenue'],
                            text=f"{growth:+.1f}%",
                            showarrow=False,
                            yshift=30,
                            font=dict(size=11, color=color, family="Arial Black"),
                            bgcolor="white",
                            bordercolor=color,
                            borderwidth=1
                        )
                
                fig_total_trend.update_layout(
                    title=f"Total Monthly Revenue - {tab_name}",
                    height=450,
                    hovermode="x unified",
                    plot_bgcolor="white",
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
                    xaxis=dict(title="Month", tickangle=-30),
                    yaxis=dict(title="Revenue ($)", gridcolor=GRID, showgrid=True)
                )
                st.plotly_chart(fig_total_trend, use_container_width=True, key=f"{tab_name}_rev_trend")
                
                # Summary stats for revenue
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    avg_monthly = revenue_pod['Revenue'].mean()
                    st.metric("Avg Monthly Revenue", _revenue_fmt(avg_monthly))
                with col2:
                    if len(revenue_pod) > 1:
                        latest_growth = revenue_pod_copy.iloc[-1]['MoM_Growth']
                        st.metric("Latest MoM Growth", f"{latest_growth:+.1f}%" if pd.notna(latest_growth) else "N/A")
                    else:
                        st.metric("Latest MoM Growth", "N/A")
                with col3:
                    max_revenue = revenue_pod['Revenue'].max()
                    max_month = revenue_pod.loc[revenue_pod['Revenue'].idxmax(), 'Month_Display']
                    st.metric("Peak Month", f"{max_month}: {_revenue_fmt(max_revenue)}")
                with col4:
                    total_period_revenue = revenue_pod['Revenue'].sum()
                    st.metric("Total Period Revenue", _revenue_fmt(total_period_revenue))
        else:
            st.info(f"Insufficient data for month-over-month analysis in {tab_name}")
    else:
        st.info(f"Revenue data (TOTAL CHARGES) not available for {tab_name} performance analysis")

    st.markdown("---")

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
        st.plotly_chart(fig, use_container_width=True, key=f"{tab_name}_net_by_vol" )
    else:
        st.info("No monthly volume available.")

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
        st.plotly_chart(figp, use_container_width=True, key=f"{tab_name}_net_by_pcs")
    else:
        st.info("No monthly PIECES available.")

    st.markdown("---")

    # ---------------- Chart: Gross vs Net OTP (POD) - MODIFIED WITH NET ON TOP ----------------
    st.subheader(f"{tab_name}: Monthly OTP Trend (Gross vs Net) — POD Month")
    if not otp_pod.empty:
        otp_sorted = otp_pod.sort_values("Month_Sort")
        x       = otp_sorted["Month_Display"].tolist()
        gross_y = otp_sorted["Gross_OTP"].astype(float).tolist()
        net_y   = otp_sorted["Net_OTP"].astype(float).tolist()

        fig2 = go.Figure()
        # Changed order - Net first (will show on top)
        fig2.add_trace(go.Scatter(x=x, y=net_y, mode="lines+markers", name="Net OTP",
                                  line=dict(color=GREEN, width=3), marker=dict(size=8)))
        fig2.add_trace(go.Scatter(x=x, y=gross_y, mode="lines+markers", name="Gross OTP",
                                  line=dict(color=BLUE, width=3), marker=dict(size=8)))
        
        # Add percentage labels for Net OTP (on top)
        for xi, yi in zip(x, net_y):
            if pd.notna(yi):
                fig2.add_annotation(
                    x=xi, y=yi, xref="x", yref="y",
                    text=f"<b>{yi:.2f}%</b>",
                    showarrow=False,
                    yshift=20,
                    font=dict(size=12, color=GREEN),
                    bgcolor="rgba(255,255,255,0.8)"
                )
        
        # Add percentage labels for Gross OTP (below)
        for xi, yi in zip(x, gross_y):
            if pd.notna(yi):
                fig2.add_annotation(
                    x=xi, y=yi, xref="x", yref="y",
                    text=f"<b>{yi:.2f}%</b>",
                    showarrow=False,
                    yshift=-20,
                    font=dict(size=12, color=BLUE),
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
        st.plotly_chart(fig2, use_container_width=True, key=f"{tab_name}_otp_trend")
    else:
        st.info("No monthly OTP trend available.")

    # ---------------- Dynamic Worst Accounts by Month ----------------
    st.subheader(f"{tab_name}: 5 Worst Accounts by Net OTP")
    
    if 'ACCT NM' in processed_df.columns:
        base = processed_df.dropna(subset=['_pod', '_target']).copy()
        if not base.empty:
            # Get unique months for selection
            base['Month_Year'] = base['_pod'].dt.to_period('M')
            unique_periods = sorted(base['Month_Year'].unique())
            
            if unique_periods:
                # Create month selector
                selected_period = st.selectbox(
                    f"Select month for worst accounts analysis ({tab_name}):",
                    options=unique_periods,
                    format_func=lambda x: x.strftime('%B %Y'),
                    index=len(unique_periods)-1,  # Default to most recent
                    key=f"{tab_name}_worst_month_select"
                )
                
                # Filter for selected month
                month_df = base[base['Month_Year'] == selected_period]
                
                if not month_df.empty:
                    # Clean account names
                    month_df['ACCT NM'] = month_df['ACCT NM'].astype(str).str.strip()
                    month_df = month_df[month_df['ACCT NM'].ne('')]
                    
                    if not month_df.empty:
                        # Net OTP (% as mean of boolean) + Volume (count of rows) for selected month
                        grp = (month_df.groupby('ACCT NM', as_index=False)
                                      .agg(Net_OTP=('On_Time_Net', 'mean'),
                                           Volume=('On_Time_Net', 'size')))
                        grp['Net_OTP'] = grp['Net_OTP'] * 100
                        
                        # Exclude NaN and 0% Net OTP
                        grp = grp[grp['Net_OTP'].notna() & (grp['Net_OTP'] > 0)]
                        
                        if not grp.empty:
                            worst = grp.nsmallest(5, 'Net_OTP').copy()
                            worst['Net_OTP'] = worst['Net_OTP'].round(2)
                            
                            # Bar chart with Net OTP and Volume in labels + hover
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
                            
                            # Target reference
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
                            st.plotly_chart(figw, use_container_width=True, key=f"{tab_name}_worst5_chart")
                            
                            # Compact table below for auditing
                            st.caption(f"Worst 5 accounts — {selected_period.strftime('%B %Y')} (Net OTP and Volume)")
                            st.dataframe(
                                worst[['ACCT NM', 'Net_OTP', 'Volume']].rename(
                                    columns={'ACCT NM':'Account', 'Net_OTP':'Net OTP (%)'}
                                ),
                                use_container_width=True,
                                key=f"{tab_name}_worst5_table"
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

    # ---------------- Optional QC breakdown ----------------
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
    
    **Data Processing:**
    - Reads **ALL SHEETS** from Excel file
    - Processes **ALL ROWS** from each sheet
    - Then filters for EMEA countries (if applicable)
    - Then filters for STATUS = 440-BILLED (if applicable)
    - Month grouping by POD DATE/TIME
    
    **New Features:**
    - Total monthly revenue trend
    - Month-over-month performance analysis
    - Enhanced tables with cross-metrics
    - Complete row tracking in debug mode
    """)

if not uploaded_file:
    st.info("""
    👆 **Please upload your Excel file to begin.**
    
    **File processing:**
    - Reads **EVERY SINGLE SHEET** from the Excel file
    - Captures **EVERY SINGLE ROW** from each sheet
    - Filters are applied ONLY if the columns exist:
      - EMEA filter: Applied only if 'PU CTRY' column exists
      - Status filter: Applied only if 'STATUS' column exists
    - Rows with missing values in filter columns are KEPT
    - Categorizes accounts as Healthcare or Non-Healthcare
    - Calculates OTP metrics by POD month
    - Analyzes month-over-month performance changes
    
    **Required columns:**
    - POD DATE/TIME (for monthly grouping)
    - ACCT NM (for account categorization)
    
    **Optional columns:**
    - PU CTRY (for EMEA filtering)
    - STATUS (for 440-BILLED filtering)
    - UPD DEL or QDT (for OTP calculations)
    - QC NAME (for controllability analysis)
    - PIECES (for volume metrics)
    - TOTAL CHARGES (for revenue analysis)
    """)
    st.stop()

# Process uploaded file
with st.spinner("Processing ALL sheets and ALL rows from Excel file..."):
    healthcare_df, non_healthcare_df, stats = read_and_combine_sheets(uploaded_file)

# ---------------- NEW: Collect ALL accounts from ALL sheets WITH FILTER REASONS ----------------
all_accounts_data = []
xls = pd.ExcelFile(uploaded_file)
for sheet_name in xls.sheet_names:
    try:
        sheet_df = pd.read_excel(xls, sheet_name=sheet_name)
        if 'ACCT NM' in sheet_df.columns:
            # For each account, track why rows are filtered out
            accounts_in_sheet = []
            
            for account_name in sheet_df['ACCT NM'].unique():
                # Get all rows for this account
                account_rows = sheet_df[sheet_df['ACCT NM'] == account_name] if pd.notna(account_name) else sheet_df[sheet_df['ACCT NM'].isna()]
                total_rows = len(account_rows)
                
                filter_reasons = []
                current_rows = account_rows.copy()
                
                # Check for missing POD DATE/TIME (required for analysis)
                missing_date_count = 0
                if 'POD DATE/TIME' in current_rows.columns:
                    missing_date_mask = current_rows['POD DATE/TIME'].isna()
                    missing_date_count = missing_date_mask.sum()
                    if missing_date_count > 0:
                        filter_reasons.append(f"Missing POD DATE ({missing_date_count} rows)")
                    # Remove rows with missing dates for further filtering
                    current_rows = current_rows[~missing_date_mask]
                else:
                    # No POD DATE column at all
                    filter_reasons.append(f"No POD DATE column ({total_rows} rows)")
                    current_rows = pd.DataFrame()  # All rows filtered
                
                # Check EMEA filter (on remaining rows)
                emea_filtered = 0
                if len(current_rows) > 0 and 'PU CTRY' in current_rows.columns:
                    current_rows_copy = current_rows.copy()
                    current_rows_copy['PU CTRY'] = current_rows_copy['PU CTRY'].astype(str).str.strip().str.upper()
                    current_rows_copy['PU CTRY'] = current_rows_copy['PU CTRY'].replace(['NAN', 'NONE', '<NA>'], '')
                    emea_mask = (
                        (current_rows_copy['PU CTRY'].isin(EMEA_COUNTRIES)) | 
                        (current_rows_copy['PU CTRY'] == '') |
                        (current_rows_copy['PU CTRY'].isna())
                    )
                    emea_filtered = (~emea_mask).sum()
                    if emea_filtered > 0:
                        filter_reasons.append(f"Not EMEA ({emea_filtered} rows)")
                    current_rows = current_rows[emea_mask]
                
                # Check STATUS filter (on remaining rows)
                status_filtered = 0
                if len(current_rows) > 0 and 'STATUS' in current_rows.columns:
                    current_rows_copy = current_rows.copy()
                    current_rows_copy['STATUS'] = current_rows_copy['STATUS'].astype(str).str.strip()
                    status_mask = (
                        (current_rows_copy['STATUS'] == '440-BILLED') |
                        (current_rows_copy['STATUS'] == '') |
                        (current_rows_copy['STATUS'] == 'nan') |
                        (current_rows_copy['STATUS'].isna())
                    )
                    status_filtered = (~status_mask).sum()
                    if status_filtered > 0:
                        filter_reasons.append(f"Not 440-BILLED ({status_filtered} rows)")
                    current_rows = current_rows[status_mask]
                
                # Check for other missing critical data in remaining rows
                if len(current_rows) > 0:
                    # Check for missing REVENUE/TOTAL CHARGES
                    if 'TOTAL CHARGES' in current_rows.columns:
                        missing_revenue = current_rows['TOTAL CHARGES'].isna().sum()
                        if missing_revenue > 0:
                            filter_reasons.append(f"Missing TOTAL CHARGES ({missing_revenue} rows)")
                    
                    # Check for missing HAWB (shipment ID)
                    if 'HAWB' in current_rows.columns:
                        missing_hawb = current_rows['HAWB'].isna().sum()
                        if missing_hawb > 0:
                            filter_reasons.append(f"Missing HAWB ({missing_hawb} rows)")
                
                rows_after_filters = len(current_rows)
                
                # Determine filter reason text
                if not filter_reasons:
                    filter_reason_text = "✓ Passed all filters"
                else:
                    filter_reason_text = "; ".join(filter_reasons)
                
                accounts_in_sheet.append({
                    'ACCT NM': account_name,
                    'Row_Count': total_rows,
                    'Rows_After_Filters': rows_after_filters,
                    'Filter_Reason': filter_reason_text,
                    'Source_Sheet': sheet_name
                })
            
            # Convert to dataframe
            accounts_df = pd.DataFrame(accounts_in_sheet)
            
            # Classify each account
            accounts_df['Classification'] = accounts_df['ACCT NM'].apply(
                lambda x: 'Healthcare' if is_healthcare(x, sheet_name) else 'Non-Healthcare'
            )
            
            all_accounts_data.append(accounts_df)
    except Exception as e:
        pass  # Skip sheets with errors

# Combine all accounts
if all_accounts_data:
    all_accounts_df = pd.concat(all_accounts_data, ignore_index=True)
    
    # Aggregate by account name across all sheets
    # Use dropna=False to include rows with missing account names
    all_accounts_summary = all_accounts_df.groupby('ACCT NM', dropna=False).agg({
        'Row_Count': 'sum',
        'Rows_After_Filters': 'sum',
        'Classification': 'first',
        'Source_Sheet': lambda x: ', '.join(sorted(set(x))),
        'Filter_Reason': lambda x: ' | '.join([r for r in set(x) if r != "✓ Passed all filters"]) or "✓ Passed all filters"
    }).reset_index()
    
    # Check which accounts are actually used in the filtered data
    used_accounts_hc = set(healthcare_df['ACCT NM'].dropna().unique()) if not healthcare_df.empty and 'ACCT NM' in healthcare_df.columns else set()
    used_accounts_nhc = set(non_healthcare_df['ACCT NM'].dropna().unique()) if not non_healthcare_df.empty and 'ACCT NM' in non_healthcare_df.columns else set()
    used_accounts = used_accounts_hc | used_accounts_nhc
    all_accounts_summary['Used_in_Filter'] = all_accounts_summary['ACCT NM'].isin(used_accounts)
    
    # Sort by classification and then by row count
    all_accounts_summary = all_accounts_summary.sort_values(
        ['Classification', 'Row_Count'], 
        ascending=[True, False]
    )
else:
    all_accounts_summary = pd.DataFrame()

# Detailed validation info - ENHANCED
if debug_mode:
    with st.expander("🔍 Debug: COMPLETE Data Flow Tracking"):
        st.write("**📊 COMPREHENSIVE ROW TRACKING:**")
        st.write(f"**TOTAL RAW ROWS (before ANY filtering):** {stats.get('total_rows_raw', 0):,}")
        st.write(f"**TOTAL COMBINED RAW ROWS:** {stats.get('total_combined_raw', 0):,}")
        
        st.write("\n**1. Sheets Read:**")
        st.write(f"   Total sheets in file: {len(stats.get('sheets_read', []))}")
        st.write(f"   Sheet names: {', '.join(stats.get('sheets_read', []))}")
        
        st.write("\n**2. DETAILED Sheet-by-Sheet Breakdown:**")
        total_raw_check = 0
        for sheet_name in stats.get('sheets_read', []):
            if sheet_name in stats.get('by_sheet', {}):
                sheet_stats = stats['by_sheet'][sheet_name]
                st.write(f"\n   **{sheet_name}:**")
                st.write(f"      - RAW ROWS (ALL DATA): {sheet_stats.get('raw_rows', 0):,} rows")
                st.write(f"      - Initial rows: {sheet_stats['initial']:,} rows")
                st.write(f"      - After EMEA filter: {sheet_stats['emea']:,} rows")
                st.write(f"      - After 440-BILLED filter: {sheet_stats['final']:,} rows")
                
                total_raw_check += sheet_stats.get('raw_rows', 0)
                
                # Calculate retention rate per sheet
                if sheet_stats.get('raw_rows', 0) > 0:
                    retention = (sheet_stats['final'] / sheet_stats.get('raw_rows', 0)) * 100
                    if retention < 50:
                        st.warning(f"      ⚠️ Low retention: {retention:.1f}% of raw data kept")
                
                if 'error' in sheet_stats:
                    st.error(f"      - Error: {sheet_stats['error']}")
        
        st.write(f"\n**✅ RAW ROW VALIDATION:**")
        st.write(f"   Sum of sheet raw rows: {total_raw_check:,}")
        st.write(f"   Total raw rows tracked: {stats.get('total_rows_raw', 0):,}")
        if total_raw_check == stats.get('total_rows_raw', 0):
            st.success("   ✅ All raw rows accounted for!")
        else:
            st.error(f"   ❌ Mismatch: {abs(total_raw_check - stats.get('total_rows_raw', 0)):,} rows difference")
        
        st.write("\n**3. Filter Impact:**")
        st.write(f"   Total rows (all sheets, raw): {stats.get('total_rows_raw', 0):,}")
        st.write(f"   After EMEA filter: {stats.get('emea_rows', 0):,} (-{stats.get('total_rows_raw', 0) - stats.get('emea_rows', 0):,} rows)")
        st.write(f"   After 440-BILLED filter: {stats.get('status_filtered', 0):,} (-{stats.get('emea_rows', 0) - stats.get('status_filtered', 0):,} rows)")
        
        if stats.get('total_rows_raw', 0) > 0:
            overall_retention = (stats.get('status_filtered', 0) / stats.get('total_rows_raw', 0)) * 100
            st.write(f"\n   **Overall Retention Rate: {overall_retention:.1f}%**")
        
        st.write("\n**4. Healthcare Classification:**")
        st.write(f"   Healthcare: {stats.get('healthcare_rows', 0):,} rows")
        st.write(f"   Non-Healthcare: {stats.get('non_healthcare_rows', 0):,} rows")
        
        # Show which sheets contribute to each category
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

# Show processing statistics - ENHANCED
with st.expander("📈 Data Processing Statistics"):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("RAW Rows (All)", f"{stats.get('total_rows_raw', 0):,}")
    with col2:
        st.metric("Total Sheets", f"{len(stats.get('sheets_read', []))}")
    with col3:
        st.metric("EMEA Filtered", f"{stats.get('emea_rows', 0):,}")
    with col4:
        st.metric("440-BILLED", f"{stats.get('status_filtered', 0):,}")
    with col5:
        st.metric("HC / Non-HC", f"{stats.get('healthcare_rows', 0):,} / {stats.get('non_healthcare_rows', 0):,}")
    
    if 'by_sheet' in stats:
        st.markdown("#### Breakdown by Sheet:")
        sheet_df = pd.DataFrame(stats['by_sheet']).T
        
        # Reorder columns to show raw rows first
        if 'raw_rows' in sheet_df.columns:
            sheet_df = sheet_df[['raw_rows', 'initial', 'emea', 'final']]
            sheet_df.columns = ['Raw Rows (ALL)', 'Initial', 'After EMEA', 'After Status']
        else:
            sheet_df.columns = ['Initial Rows', 'After EMEA Filter', 'After Status Filter']
        
        st.dataframe(sheet_df)
        
        # Additional validation
        st.markdown("#### Validation:")
        st.write(f"✓ Sheets processed: {len(stats.get('sheets_read', []))}")
        st.write(f"✓ Raw rows (all sheets): {stats.get('total_rows_raw', 0):,}")
        st.write(f"✓ Rows after all filters: {stats.get('status_filtered', 0):,}")
        total_processed = stats.get('healthcare_rows', 0) + stats.get('non_healthcare_rows', 0)
        st.write(f"✓ HC + Non-HC total: {total_processed:,}")
        if total_processed == stats.get('status_filtered', 0):
            st.success("✅ All filtered rows are categorized correctly!")
        else:
            st.warning(f"⚠️ Mismatch: {stats.get('status_filtered', 0) - total_processed} rows not categorized")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🏥 Healthcare", "✈️ Non-Healthcare", "📋 All Accounts Overview"])

with tab1:
    st.markdown("## Healthcare Sector Analysis")
    if not healthcare_df.empty:
        st.markdown(f"**Total Healthcare Entries:** {len(healthcare_df):,}")
        # Show sample accounts
        with st.expander("Sample Healthcare Accounts"):
            if 'ACCT NM' in healthcare_df.columns:
                unique_accounts = healthcare_df['ACCT NM'].dropna().unique()[:20]
                st.write(", ".join(unique_accounts))
        
        # Debug info
        if debug_mode:
            with st.expander("🔍 Debug: Healthcare POD Date Processing"):
                if 'POD DATE/TIME' in healthcare_df.columns:
                    sample_pod = healthcare_df[['POD DATE/TIME']].dropna().head(10)
                    st.write("Sample POD DATE/TIME values:")
                    st.dataframe(sample_pod)
                    
                    # Show how dates are being parsed
                    test_dates = _excel_to_dt(healthcare_df['POD DATE/TIME'].head(10))
                    st.write("Parsed dates:")
                    st.write(test_dates.to_list())
                    
                    # Show month grouping
                    st.write("Month grouping (YYYY-MM):")
                    st.write(test_dates.dt.to_period("M").astype(str).to_list())
    
    create_dashboard_view(healthcare_df, "Healthcare", otp_target, debug_mode)

with tab2:
    st.markdown("## Non-Healthcare Sector Analysis")
    if not non_healthcare_df.empty:
        st.markdown(f"**Total Non-Healthcare Entries:** {len(non_healthcare_df):,}")
        # Show sample accounts
        with st.expander("Sample Non-Healthcare Accounts"):
            if 'ACCT NM' in non_healthcare_df.columns:
                unique_accounts = non_healthcare_df['ACCT NM'].dropna().unique()[:20]
                st.write(", ".join(unique_accounts))
        
        # Debug info
        if debug_mode:
            with st.expander("🔍 Debug: Non-Healthcare POD Date Processing"):
                if 'POD DATE/TIME' in non_healthcare_df.columns:
                    sample_pod = non_healthcare_df[['POD DATE/TIME']].dropna().head(10)
                    st.write("Sample POD DATE/TIME values:")
                    st.dataframe(sample_pod)
                    
                    # Show how dates are being parsed
                    test_dates = _excel_to_dt(non_healthcare_df['POD DATE/TIME'].head(10))
                    st.write("Parsed dates:")
                    st.write(test_dates.to_list())
                    
                    # Show month grouping
                    st.write("Month grouping (YYYY-MM):")
                    st.write(test_dates.dt.to_period("M").astype(str).to_list())
    
    create_dashboard_view(non_healthcare_df, "Non-Healthcare", otp_target, debug_mode)

# NEW TAB: All Accounts Overview
with tab3:
    st.markdown("## 📋 All Accounts Overview")
    st.markdown("""
    This tab shows **ALL accounts** from **ALL sheets** in the Excel file with complete filtering details:
    - 🟢 **Green**: Accounts classified as **Healthcare** and used in filtered data
    - 🔵 **Blue**: Accounts classified as **Non-Healthcare** and used in filtered data
    - ⚪ **No highlight**: Accounts that exist but are **not used** - see "Why Filtered Out?" for detailed reasons
    
    **Filter Reasons Include:**
    - ✓ Passed all filters
    - ❌ Missing POD DATE (required for analysis)
    - ❌ Not EMEA (country filter)
    - ❌ Not 440-BILLED (status filter)
    - ❌ Missing TOTAL CHARGES or HAWB (data quality issues)
    """)
    
    if not all_accounts_summary.empty:
        # Summary metrics
        total_accounts = len(all_accounts_summary)
        hc_accounts = len(all_accounts_summary[all_accounts_summary['Classification'] == 'Healthcare'])
        non_hc_accounts = len(all_accounts_summary[all_accounts_summary['Classification'] == 'Non-Healthcare'])
        used_accounts = len(all_accounts_summary[all_accounts_summary['Used_in_Filter'] == True])
        unused_accounts = len(all_accounts_summary[all_accounts_summary['Used_in_Filter'] == False])
        total_rows_all_accounts = all_accounts_summary['Row_Count'].sum()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Accounts", f"{total_accounts:,}")
        with col2:
            st.metric("Healthcare", f"{hc_accounts:,}", delta=None, delta_color="normal")
        with col3:
            st.metric("Non-Healthcare", f"{non_hc_accounts:,}", delta=None, delta_color="normal")
        with col4:
            st.metric("Used in Filter", f"{used_accounts:,}", delta=None, delta_color="normal")
        with col5:
            st.metric("Filtered Out", f"{unused_accounts:,}", delta=None, delta_color="inverse")
        
        # Validation info
        st.info(f"📊 **Total rows across all accounts:** {total_rows_all_accounts:,} rows from all sheets (including rows with missing account names)")
        
        st.markdown("---")
        
        # Filter options
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            classification_filter = st.multiselect(
                "Filter by Classification:",
                options=['Healthcare', 'Non-Healthcare'],
                default=['Healthcare', 'Non-Healthcare']
            )
        with col_filter2:
            usage_filter = st.radio(
                "Filter by Usage:",
                options=['All', 'Used Only', 'Unused Only'],
                horizontal=True
            )
        
        # Apply filters
        filtered_accounts = all_accounts_summary.copy()
        if classification_filter:
            filtered_accounts = filtered_accounts[filtered_accounts['Classification'].isin(classification_filter)]
        
        if usage_filter == 'Used Only':
            filtered_accounts = filtered_accounts[filtered_accounts['Used_in_Filter'] == True]
        elif usage_filter == 'Unused Only':
            filtered_accounts = filtered_accounts[filtered_accounts['Used_in_Filter'] == False]
        
        st.markdown(f"**Showing {len(filtered_accounts):,} of {total_accounts:,} accounts**")
        
        # Prepare display dataframe with formatting
        display_df = filtered_accounts.copy().reset_index(drop=True)
        
        # Store original values for highlighting before formatting
        original_used = display_df['Used_in_Filter'].copy()
        original_classification = display_df['Classification'].copy()
        
        # Format the display values
        display_df['Row_Count'] = display_df['Row_Count'].apply(lambda x: f"{x:,}")
        display_df['Rows_After_Filters'] = display_df['Rows_After_Filters'].apply(lambda x: f"{x:,}")
        display_df['Used_in_Filter'] = display_df['Used_in_Filter'].map({True: '✓ Yes', False: '✗ No'})
        
        # Reorder and rename columns to show filter reason
        display_df = display_df[[
            'ACCT NM', 'Row_Count', 'Rows_After_Filters', 'Classification', 
            'Source_Sheet', 'Used_in_Filter', 'Filter_Reason'
        ]]
        
        display_df = display_df.rename(columns={
            'ACCT NM': 'Account Name',
            'Row_Count': 'Total Rows',
            'Rows_After_Filters': 'Rows After Filters',
            'Classification': 'Classification',
            'Source_Sheet': 'Found in Sheets',
            'Used_in_Filter': 'Used?',
            'Filter_Reason': 'Why Filtered Out?'
        })
        
        # Create styling function that uses indices to access original values
        def highlight_accounts(row):
            idx = row.name
            if not original_used.iloc[idx]:
                return [''] * len(row)  # No highlighting for unused accounts
            elif original_classification.iloc[idx] == 'Healthcare':
                return ['background-color: #d1fae5'] * len(row)  # Light green
            else:
                return ['background-color: #dbeafe'] * len(row)  # Light blue
        
        st.dataframe(display_df.style.apply(highlight_accounts, axis=1), use_container_width=True, height=600)
        
        # Detailed breakdown
        with st.expander("📊 Detailed Account Analysis"):
            st.markdown("#### Accounts Used in Filtered Data:")
            used_df = all_accounts_summary[all_accounts_summary['Used_in_Filter'] == True].copy()
            if not used_df.empty:
                hc_used = len(used_df[used_df['Classification'] == 'Healthcare'])
                non_hc_used = len(used_df[used_df['Classification'] == 'Non-Healthcare'])
                st.write(f"- **Healthcare accounts used:** {hc_used:,}")
                st.write(f"- **Non-Healthcare accounts used:** {non_hc_used:,}")
                st.write(f"- **Total rows in filtered data:** {used_df['Rows_After_Filters'].sum():,}")
            
            st.markdown("#### Accounts Filtered Out (Not Used):")
            unused_df = all_accounts_summary[all_accounts_summary['Used_in_Filter'] == False].copy()
            if not unused_df.empty:
                hc_unused = len(unused_df[unused_df['Classification'] == 'Healthcare'])
                non_hc_unused = len(unused_df[unused_df['Classification'] == 'Non-Healthcare'])
                total_filtered_rows = unused_df['Row_Count'].sum()
                total_surviving_rows = unused_df['Rows_After_Filters'].sum()
                
                st.write(f"- **Healthcare accounts filtered out:** {hc_unused:,}")
                st.write(f"- **Non-Healthcare accounts filtered out:** {non_hc_unused:,}")
                st.write(f"- **Total rows from these accounts:** {total_filtered_rows:,}")
                st.write(f"- **Rows that passed filters:** {total_surviving_rows:,}")
                st.write(f"- **Rows actually filtered out:** {total_filtered_rows - total_surviving_rows:,}")
                
                # Analyze filter reasons
                st.markdown("#### Why Accounts Were Filtered Out:")
                
                # Count by filter reason type
                missing_date_count = unused_df[unused_df['Filter_Reason'].str.contains('Missing POD DATE|No POD DATE', na=False)].shape[0]
                emea_count = unused_df[unused_df['Filter_Reason'].str.contains('Not EMEA', na=False)].shape[0]
                status_count = unused_df[unused_df['Filter_Reason'].str.contains('Not 440-BILLED', na=False)].shape[0]
                missing_revenue_count = unused_df[unused_df['Filter_Reason'].str.contains('Missing TOTAL CHARGES', na=False)].shape[0]
                missing_hawb_count = unused_df[unused_df['Filter_Reason'].str.contains('Missing HAWB', na=False)].shape[0]
                
                if missing_date_count > 0:
                    st.write(f"   - **Missing POD DATE (required):** {missing_date_count:,} accounts")
                if emea_count > 0:
                    st.write(f"   - **Not in EMEA countries:** {emea_count:,} accounts")
                if status_count > 0:
                    st.write(f"   - **Not 440-BILLED status:** {status_count:,} accounts")
                if missing_revenue_count > 0:
                    st.write(f"   - **Missing TOTAL CHARGES:** {missing_revenue_count:,} accounts")
                if missing_hawb_count > 0:
                    st.write(f"   - **Missing HAWB:** {missing_hawb_count:,} accounts")
                
                st.info("💡 See the 'Why Filtered Out?' column in the table above for detailed reasons per account")
        
        # Show sample accounts from each category
        with st.expander("🔍 Sample Accounts by Category"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Healthcare Accounts (Used):**")
                hc_used_sample = all_accounts_summary[
                    (all_accounts_summary['Classification'] == 'Healthcare') & 
                    (all_accounts_summary['Used_in_Filter'] == True)
                ]['ACCT NM'].head(10).tolist()
                if hc_used_sample:
                    for acc in hc_used_sample:
                        st.write(f"🟢 {acc}")
                else:
                    st.write("No healthcare accounts used")
                
                st.markdown("**Healthcare Accounts (Filtered Out):**")
                hc_unused_sample = all_accounts_summary[
                    (all_accounts_summary['Classification'] == 'Healthcare') & 
                    (all_accounts_summary['Used_in_Filter'] == False)
                ].head(5)
                if not hc_unused_sample.empty:
                    for _, row in hc_unused_sample.iterrows():
                        st.write(f"⚪ {row['ACCT NM']}")
                        st.caption(f"   └─ {row['Filter_Reason']}")
                else:
                    st.write("No healthcare accounts filtered out")
            
            with col2:
                st.markdown("**Non-Healthcare Accounts (Used):**")
                non_hc_used_sample = all_accounts_summary[
                    (all_accounts_summary['Classification'] == 'Non-Healthcare') & 
                    (all_accounts_summary['Used_in_Filter'] == True)
                ]['ACCT NM'].head(10).tolist()
                if non_hc_used_sample:
                    for acc in non_hc_used_sample:
                        st.write(f"🔵 {acc}")
                else:
                    st.write("No non-healthcare accounts used")
                
                st.markdown("**Non-Healthcare Accounts (Filtered Out):**")
                non_hc_unused_sample = all_accounts_summary[
                    (all_accounts_summary['Classification'] == 'Non-Healthcare') & 
                    (all_accounts_summary['Used_in_Filter'] == False)
                ].head(5)
                if not non_hc_unused_sample.empty:
                    for _, row in non_hc_unused_sample.iterrows():
                        st.write(f"⚪ {row['ACCT NM']}")
                        st.caption(f"   └─ {row['Filter_Reason']}")
                else:
                    st.write("No non-healthcare accounts filtered out")
    else:
        st.warning("No account data available")
