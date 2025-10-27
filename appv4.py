#Code 1 - Complete Version - ALL ROWS GUARANTEED - CORRECTED
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
    """Convert Excel serial dates and datetime strings to pandas datetime."""
    if s.dtype == 'datetime64[ns]':
        return s
    numeric_mask = pd.to_numeric(s, errors='coerce').notna()
    result = pd.Series(pd.NaT, index=s.index)
    if numeric_mask.any():
        numeric_vals = pd.to_numeric(s[numeric_mask], errors='coerce')
        result[numeric_mask] = pd.to_datetime('1900-01-01') + pd.to_timedelta(numeric_vals - 2, unit='D')
    string_mask = ~numeric_mask & s.notna()
    if string_mask.any():
        result[string_mask] = pd.to_datetime(s[string_mask], errors='coerce')
    return result

def _get_target_series(d: pd.DataFrame) -> pd.Series:
    """Find the target date column."""
    for col in ["UPD DEL", "ORG DEL", "QDT"]:
        if col in d.columns:
            return d[col]
    return None

def _kfmt(v: float) -> str:
    """Format numbers with K/M suffix."""
    if pd.isna(v):
        return "N/A"
    if v >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    elif v >= 10_000:
        return f"{v/1_000:.0f}K"
    elif v >= 1_000:
        return f"{v/1_000:.1f}K"
    else:
        return str(int(v))

def _revenue_fmt(v: float) -> str:
    """Format revenue values with currency symbol."""
    if pd.isna(v):
        return "N/A"
    if v >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    elif v >= 1_000:
        return f"${v/1_000:.1f}K"
    else:
        return f"${v:.0f}"

def is_healthcare(account_name: str, sheet_name: str = "") -> bool:
    """Determine if an account is healthcare-related."""
    if pd.isna(account_name):
        return False
    
    account_lower = str(account_name).lower()
    sheet_lower = str(sheet_name).lower()
    
    # Sheet-level classification (highest priority)
    if 'non-healthcare' in sheet_lower or 'nonhealthcare' in sheet_lower:
        return False
    if 'healthcare' in sheet_lower:
        return True
    
    # Check for explicit exclusions first
    for keyword in NON_HEALTHCARE_KEYWORDS:
        if keyword in account_lower:
            return False
    
    # Check for healthcare keywords
    for keyword in HEALTHCARE_KEYWORDS:
        if keyword in account_lower:
            return True
    
    return False

# ---------------- File Processing ----------------
@st.cache_data(show_spinner=False)
def read_excel_sheets(uploaded) -> list:
    """Read all sheet names from uploaded Excel file."""
    try:
        xl = pd.ExcelFile(uploaded, engine='openpyxl')
        return xl.sheet_names
    except Exception as e:
        st.error(f"Error reading Excel file: {str(e)}")
        return []

@st.cache_data(show_spinner=False)
def collect_all_accounts(uploaded) -> pd.DataFrame:
    """Collect all accounts from all sheets for overview."""
    try:
        all_sheet_names = read_excel_sheets(uploaded)
        all_accounts_data = []
        
        for sheet_name in all_sheet_names:
            try:
                df_sheet = pd.read_excel(uploaded, sheet_name=sheet_name, engine='openpyxl')
                if df_sheet.empty or 'ACCT NM' not in df_sheet.columns:
                    continue
                
                accounts = df_sheet['ACCT NM'].dropna()
                if accounts.empty:
                    continue
                
                # Get unique accounts with row counts
                account_counts = accounts.value_counts()
                for account_name, count in account_counts.items():
                    all_accounts_data.append({
                        'ACCT NM': account_name,
                        'Row_Count': count,
                        'Source_Sheet': sheet_name
                    })
            except:
                continue
        
        if all_accounts_data:
            all_accounts_df = pd.DataFrame(all_accounts_data)
            
            # Aggregate by account name across all sheets
            all_accounts_summary = all_accounts_df.groupby('ACCT NM', dropna=False).agg({
                'Row_Count': 'sum',
                'Source_Sheet': lambda x: ', '.join(sorted(set(x)))
            }).reset_index()
            
            # Classify each account
            all_accounts_summary['Classification'] = all_accounts_summary.apply(
                lambda row: 'Healthcare' if is_healthcare(row['ACCT NM'], row['Source_Sheet']) else 'Non-Healthcare',
                axis=1
            )
            
            return all_accounts_summary
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error collecting accounts: {str(e)}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def read_and_combine_sheets(uploaded):
    """Read all sheets and combine Healthcare and Non-Healthcare data with filters."""
    try:
        all_sheet_names = read_excel_sheets(uploaded)
        if not all_sheet_names:
            st.error("No sheets found in the Excel file")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
        
        all_data_filtered = []
        all_data_emea = []  # EMEA-only data (no STATUS filter) for gross charts
        
        stats = {
            'total_rows_raw': 0,
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
                
                # Store EMEA data BEFORE STATUS filter for gross calculations
                if len(df_sheet_emea) > 0:
                    all_data_emea.append(df_sheet_emea.copy())
                
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
        healthcare_df_gross = pd.DataFrame()
        non_healthcare_df_gross = pd.DataFrame()
        
        if not combined_df.empty and 'ACCT NM' in combined_df.columns:
            # Apply healthcare classification with sheet context
            combined_df['Is_Healthcare'] = combined_df.apply(
                lambda row: is_healthcare(row.get('ACCT NM', ''), row.get('Source_Sheet', '')), axis=1
            )
            
            healthcare_df = combined_df[combined_df['Is_Healthcare'] == True].copy()
            non_healthcare_df = combined_df[combined_df['Is_Healthcare'] == False].copy()
            
            stats['healthcare_rows'] = len(healthcare_df)
            stats['non_healthcare_rows'] = len(non_healthcare_df)
            
            # Also create EMEA-only dataframes (without STATUS filter) for gross charts
            if all_data_emea:
                combined_df_gross = pd.concat(all_data_emea, ignore_index=True)
                # Apply healthcare classification
                combined_df_gross['Is_Healthcare'] = combined_df_gross.apply(
                    lambda row: is_healthcare(row.get('ACCT NM', ''), row.get('Source_Sheet', '')), axis=1
                )
                healthcare_df_gross = combined_df_gross[combined_df_gross['Is_Healthcare'] == True].copy()
                non_healthcare_df_gross = combined_df_gross[combined_df_gross['Is_Healthcare'] == False].copy()
        
        return healthcare_df, non_healthcare_df, healthcare_df_gross, non_healthcare_df_gross, stats
    
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

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

def create_dashboard_view(df: pd.DataFrame, tab_name: str, otp_target: float, debug_mode: bool = False, gross_df: pd.DataFrame = None):
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
                st.write("**Sample month groupings:**")
                sample_months = pod_dates[['_pod', 'Month_YYYY_MM', 'Month_Display']].head(10)
                st.dataframe(sample_months)
                
                # Show month distribution
                month_counts = pod_dates['Month_Display'].value_counts().sort_index()
                st.write("**Month distribution:**")
                for month, count in month_counts.items():
                    st.write(f"  - {month}: {count:,} rows")
    
    # Get monthly breakdowns (all by POD)
    vol_pod, pieces_pod, otp_pod, revenue_pod = monthly_frames(processed_df)
    
    # Get summary
    gross_otp, net_otp, volume_total, exceptions, controllables, uncontrollables, total_revenue = calc_summary(processed_df)
    
    # KPIs
    st.markdown("### 📊 Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        delta = net_otp - otp_target if pd.notna(net_otp) else None
        color = "normal" if delta and delta >= 0 else "inverse"
        st.metric("Net OTP", f"{net_otp:.2f}%" if pd.notna(net_otp) else "N/A", 
                  delta=f"{delta:.1f}%" if delta else None, delta_color=color)
    with col2:
        st.metric("Gross OTP", f"{gross_otp:.2f}%" if pd.notna(gross_otp) else "N/A")
    with col3:
        st.metric("Total Volume", f"{volume_total:,}")
    with col4:
        st.metric("Exceptions", f"{exceptions:,}")
    with col5:
        st.metric("Total Revenue", _revenue_fmt(total_revenue))
    
    # Add controllable breakdown
    with st.expander("📈 Exception Analysis"):
        if exceptions > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Controllable Exceptions", f"{controllables:,}", 
                         help="Exceptions due to controllable factors")
            with col2:
                st.metric("Uncontrollable Exceptions", f"{uncontrollables:,}", 
                         help="Exceptions due to external factors")
            with col3:
                if exceptions > 0:
                    controllable_pct = (controllables / exceptions) * 100
                    st.metric("Controllable %", f"{controllable_pct:.1f}%")
                else:
                    st.metric("Controllable %", "N/A")
    
    st.markdown("---")
    
    # Revenue Trend Analysis (if TOTAL CHARGES column exists)
    if 'TOTAL CHARGES' in processed_df.columns and not revenue_pod.empty:
        st.subheader(f"💰 {tab_name}: Revenue Analysis — POD Month")
        
        if len(revenue_pod) > 1:
            # Create the combined bar chart with trend line
            fig_total_trend = go.Figure()
            
            # Add bar chart
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

    # ---------------- NEW: Gross Volume & Gross Pieces Charts (EMEA, All STATUS) ----------------
    st.subheader(f"📊 {tab_name}: Gross Metrics (EMEA, All STATUS)")
    st.info("💡 These charts show EMEA data with ALL STATUS values (not just 440-BILLED)")
    
    # Use the gross_df that was passed in (EMEA-only, no STATUS filter)
    if gross_df is not None and not gross_df.empty:
        # Process gross data for monthly grouping
        gross_processed = preprocess(gross_df)
        
        if not gross_processed.empty:
            # Get monthly volumes and pieces using the same logic as Net OTP charts
            gross_vol_pod, gross_pieces_pod, _, _ = monthly_frames(gross_processed)
            
            # Create a two-column layout for the gross charts
            col1, col2 = st.columns(2)
            
            # Chart 1: Gross Volume
            with col1:
                st.markdown("#### 📦 Gross Volume Month-over-Month")
                if not gross_vol_pod.empty:
                    fig_gross_vol = go.Figure()
                    fig_gross_vol.add_trace(go.Bar(
                        x=gross_vol_pod['Month_Display'],
                        y=gross_vol_pod['Volume'],
                        name='Gross Volume',
                        marker_color='#3b82f6',  # Blue
                        text=[f"{int(v):,}" for v in gross_vol_pod['Volume']],
                        textposition='outside',
                        textfont=dict(size=11, color='#1e40af', family="Arial Black")
                    ))
                    
                    fig_gross_vol.update_layout(
                        title=f"Gross Volume - {tab_name} (All STATUS)",
                        height=400,
                        hovermode="x unified",
                        plot_bgcolor="white",
                        xaxis=dict(title="Month", tickangle=-45, automargin=True),
                        yaxis=dict(title="Volume (Rows)", gridcolor=GRID, showgrid=True),
                        showlegend=False,
                        margin=dict(l=60, r=30, t=60, b=80)
                    )
                    st.plotly_chart(fig_gross_vol, use_container_width=True, key=f"{tab_name}_gross_vol")
                    
                    # Summary stats
                    st.markdown("**Summary:**")
                    total_gross_vol = gross_vol_pod['Volume'].sum()
                    avg_gross_vol = gross_vol_pod['Volume'].mean()
                    peak_month = gross_vol_pod.loc[gross_vol_pod['Volume'].idxmax(), 'Month_Display']
                    peak_vol = gross_vol_pod['Volume'].max()
                    st.write(f"- Total: {total_gross_vol:,}")
                    st.write(f"- Avg/Month: {avg_gross_vol:,.0f}")
                    st.write(f"- Peak: {peak_month} ({peak_vol:,})")
                else:
                    st.info("No gross volume data available")
            
            # Chart 2: Gross Pieces
            with col2:
                st.markdown("#### 📦 Gross Pieces Month-over-Month")
                if not gross_pieces_pod.empty:
                    fig_gross_pieces = go.Figure()
                    fig_gross_pieces.add_trace(go.Bar(
                        x=gross_pieces_pod['Month_Display'],
                        y=gross_pieces_pod['Pieces'],
                        name='Gross Pieces',
                        marker_color='#10b981',  # Green
                        text=[f"{int(v):,}" for v in gross_pieces_pod['Pieces']],
                        textposition='outside',
                        textfont=dict(size=11, color='#047857', family="Arial Black")
                    ))
                    
                    fig_gross_pieces.update_layout(
                        title=f"Gross Pieces - {tab_name} (All STATUS)",
                        height=400,
                        hovermode="x unified",
                        plot_bgcolor="white",
                        xaxis=dict(title="Month", tickangle=-45, automargin=True),
                        yaxis=dict(title="Pieces", gridcolor=GRID, showgrid=True),
                        showlegend=False,
                        margin=dict(l=60, r=30, t=60, b=80)
                    )
                    st.plotly_chart(fig_gross_pieces, use_container_width=True, key=f"{tab_name}_gross_pieces")
                    
                    # Summary stats
                    st.markdown("**Summary:**")
                    total_gross_pieces = gross_pieces_pod['Pieces'].sum()
                    avg_gross_pieces = gross_pieces_pod['Pieces'].mean()
                    peak_month = gross_pieces_pod.loc[gross_pieces_pod['Pieces'].idxmax(), 'Month_Display']
                    peak_pieces = gross_pieces_pod['Pieces'].max()
                    st.write(f"- Total: {total_gross_pieces:,.0f}")
                    st.write(f"- Avg/Month: {avg_gross_pieces:,.0f}")
                    st.write(f"- Peak: {peak_month} ({peak_pieces:,.0f})")
                else:
                    st.info("No gross pieces data available (PIECES column may be missing)")
        else:
            st.info("No data with valid POD dates for gross calculations")
    else:
        st.info("No EMEA data available for gross calculations")

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
        st.plotly_chart(figp, use_container_width=True, key=f"{tab_name}_net_by_pieces")
    else:
        st.info("No monthly pieces available.")

# ---------------- Main Execution ----------------
# File uploader
uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

# Add debug mode toggle in sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    otp_target = st.slider("OTP Target (%)", min_value=85, max_value=100, value=OTP_TARGET, step=1)
    debug_mode = st.checkbox("🔍 Enable Debug Mode", value=False)

if uploaded_file:
    # Process data
    with st.spinner('Processing Excel file...'):
        healthcare_df, non_healthcare_df, healthcare_df_gross, non_healthcare_df_gross, stats = read_and_combine_sheets(uploaded_file)
        all_accounts_summary = collect_all_accounts(uploaded_file)

    # Show enhanced debug information
    if debug_mode:
        with st.expander("🔍 Debug: Data Loading & Processing Details", expanded=True):
            st.write("### 1. File Information:")
            st.write(f"   File name: {uploaded_file.name}")
            st.write(f"   Sheets found: {len(stats.get('sheets_read', []))}")
            st.write(f"   Sheet names: {', '.join(stats.get('sheets_read', []))}")
            
            st.write("\n### 2. Row Counts by Sheet:")
            if 'by_sheet' in stats:
                total_raw_check = 0
                for sheet_name, sheet_stats in stats['by_sheet'].items():
                    raw_rows = sheet_stats.get('raw_rows', 0)
                    total_raw_check += raw_rows
                    emea = sheet_stats.get('emea', 0)
                    final = sheet_stats.get('final', 0)
                    st.write(f"   **{sheet_name}:**")
                    st.write(f"      - Raw rows (ALL): {raw_rows:,}")
                    st.write(f"      - After EMEA filter: {emea:,} (-{raw_rows - emea:,})")
                    st.write(f"      - After STATUS filter: {final:,} (-{emea - final:,})")
                    
                    if 'error' in sheet_stats:
                        st.error(f"      - Error: {sheet_stats['error']}")
            
            st.write(f"\n**✅ RAW ROW VALIDATION:**")
            st.write(f"   Sum of sheet raw rows: {total_raw_check:,}")
            st.write(f"   Total raw rows tracked: {stats.get('total_rows_raw', 0):,}")
            if total_raw_check == stats.get('total_rows_raw', 0):
                st.success("   ✅ All raw rows accounted for!")
            else:
                st.error(f"   ❌ Mismatch: {abs(total_raw_check - stats.get('total_rows_raw', 0)):,} rows difference")
            
            st.write("\n### 3. Filter Impact:")
            st.write(f"   Total rows (all sheets, raw): {stats.get('total_rows_raw', 0):,}")
            st.write(f"   After EMEA filter: {stats.get('emea_rows', 0):,} (-{stats.get('total_rows_raw', 0) - stats.get('emea_rows', 0):,} rows)")
            st.write(f"   After 440-BILLED filter: {stats.get('status_filtered', 0):,} (-{stats.get('emea_rows', 0) - stats.get('status_filtered', 0):,} rows)")
            
            if stats.get('total_rows_raw', 0) > 0:
                overall_retention = (stats.get('status_filtered', 0) / stats.get('total_rows_raw', 0)) * 100
                st.write(f"\n   **Overall Retention Rate: {overall_retention:.1f}%**")
            
            st.write("\n### 4. Healthcare Classification:")
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
    tab1, tab2 = st.tabs(["🏥 Healthcare", "✈️ Non-Healthcare"])

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
        
        create_dashboard_view(healthcare_df, "Healthcare", otp_target, debug_mode, healthcare_df_gross)

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
        
        create_dashboard_view(non_healthcare_df, "Non-Healthcare", otp_target, debug_mode, non_healthcare_df_gross)

else:
    st.info("👆 Please upload an Excel file to begin.")
    
    # Show expected format
    with st.expander("📋 Expected File Format"):
        st.markdown("""
        The Excel file should contain the following columns:
        - **ACCT NM**: Account name (used for healthcare classification)
        - **POD DATE/TIME**: Delivery date/time (required)
        - **UPD DEL** or **ORG DEL** or **QDT**: Target delivery date
        - **PU CTRY**: Pickup country (for EMEA filtering)
        - **STATUS**: Status code (filter for "440-BILLED")
        - **QC NAME**: Quality control name (for controllability)
        - **PIECES**: Number of pieces (optional)
        - **TOTAL CHARGES**: Revenue amount (optional)
        - **HAWB**: House airway bill number (optional)
        
        Multiple sheets are supported and will be combined automatically.
        """)
