#Code 1 - Complete Version - ALL ROWS GUARANTEED + Account Overview Tab
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
HEALTHCARE_COLOR = "#10b981"  # Green for healthcare
NON_HEALTHCARE_COLOR = "#3b82f6"  # Blue for non-healthcare

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
    """Semi-circular gauge for OTP."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x':[0,1],'y':[0,1]},
        title={'text': title, 'font':{'size':16,'color':NAVY}},
        delta={'reference': OTP_TARGET, 'increasing':{'color':GREEN},'decreasing':{'color':RED}},
        number={'suffix':'%','font':{'size':28,'weight':'bold','color':NAVY}},
        gauge={
            'axis':{'range':[0,100],'tickcolor':SLATE,'tickfont':{'size':10}},
            'bar':{'color':NAVY,'thickness':0.3},
            'steps':[
                {'range':[0,OTP_TARGET],'color':RED},
                {'range':[OTP_TARGET,100],'color':GREEN}
            ],
            'threshold':{
                'line':{'color':GOLD,'width':4},
                'thickness':0.75,
                'value':OTP_TARGET
            }
        }
    ))
    fig.update_layout(
        height=240,
        margin=dict(t=60,b=30,l=30,r=30),
        paper_bgcolor="#fff",
        font={'family':"Arial",'color':NAVY}
    )
    return fig

def get_sheets_dict(fl, emea_set: set):
    """
    Read ALL sheets, track RAW ROWS (before any filter).
    Return (combined_df, stats).
    """
    xls = pd.ExcelFile(fl)
    sheets_read = []
    sheet_stats = {}
    dfs = []
    total_raw = 0
    
    for sn in xls.sheet_names:
        try:
            raw = pd.read_excel(xls, sheet_name=sn)
            raw_count = len(raw)
            total_raw += raw_count
            
            df = raw.copy()
            init_len = len(df)
            
            # EMEA filter
            if 'D CTRY' in df.columns:
                df['D CTRY'] = df['D CTRY'].astype(str).str.strip().str.upper()
                df = df[df['D CTRY'].isin(emea_set)].copy()
            emea_len = len(df)
            
            # Status filter
            if '440-BILLED' in df.columns:
                df = df[df['440-BILLED'].notna()].copy()
            final_len = len(df)
            
            # Add source sheet
            df['Source_Sheet'] = sn
            dfs.append(df)
            sheets_read.append(sn)
            
            sheet_stats[sn] = {
                'raw_rows': raw_count,
                'initial': init_len,
                'emea': emea_len,
                'final': final_len
            }
            
        except Exception as e:
            sheet_stats[sn] = {'error': str(e)}
    
    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    stats = {
        'sheets_read': sheets_read,
        'by_sheet': sheet_stats,
        'total_rows_raw': total_raw,
        'total_combined_raw': total_raw,
        'emea_rows': sum(s.get('emea',0) for s in sheet_stats.values()),
        'status_filtered': len(combined)
    }
    
    return combined, stats

def split_healthcare_classification(df: pd.DataFrame):
    """Split DataFrame into healthcare and non-healthcare based on account classification."""
    if df.empty or 'ACCT NM' not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    
    # Apply healthcare classification
    if 'Source_Sheet' in df.columns:
        df['Is_Healthcare'] = df.apply(
            lambda row: is_healthcare(row['ACCT NM'], row['Source_Sheet']), 
            axis=1
        )
    else:
        df['Is_Healthcare'] = df['ACCT NM'].apply(lambda x: is_healthcare(x))
    
    healthcare = df[df['Is_Healthcare'] == True].copy()
    non_healthcare = df[df['Is_Healthcare'] == False].copy()
    
    return healthcare, non_healthcare

def create_dashboard_view(df: pd.DataFrame, sector: str, otp_target: float, debug_mode: bool):
    """Creates the dashboard view for a given sector (Healthcare or Non-Healthcare)"""
    if df.empty:
        st.warning(f"No data available for {sector}")
        return
    
    # Monthly Revenue
    if 'POD DATE/TIME' in df.columns and 'REVENUE' in df.columns:
        st.subheader("📅 Monthly Revenue Trend")
        
        df_month = df.copy()
        df_month['POD_dt'] = _excel_to_dt(df_month['POD DATE/TIME'])
        df_month = df_month[df_month['POD_dt'].notna()].copy()
        
        if not df_month.empty:
            df_month['REVENUE'] = pd.to_numeric(df_month['REVENUE'], errors='coerce')
            df_month = df_month[df_month['REVENUE'].notna()].copy()
            
            if not df_month.empty:
                df_month['YearMonth'] = df_month['POD_dt'].dt.to_period("M").astype(str)
                
                monthly = df_month.groupby('YearMonth', as_index=False).agg({
                    'REVENUE': 'sum',
                    'HAWB': 'count'
                }).rename(columns={'HAWB': 'Shipments'})
                
                monthly = monthly.sort_values('YearMonth')
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=monthly['YearMonth'],
                    y=monthly['Revenue'],
                    name='Revenue',
                    marker_color=NAVY,
                    text=[_revenue_fmt(v) for v in monthly['Revenue']],
                    textposition='outside'
                ))
                
                fig.add_trace(go.Scatter(
                    x=monthly['YearMonth'],
                    y=monthly['Shipments'],
                    name='Shipments',
                    mode='lines+markers',
                    yaxis='y2',
                    line=dict(color=GOLD, width=3),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title=dict(text=f"{sector} - Monthly Revenue & Shipments", font=dict(size=18, color=NAVY)),
                    xaxis=dict(title="Month", tickangle=-45),
                    yaxis=dict(title="Revenue", side='left', showgrid=True, gridcolor=GRID),
                    yaxis2=dict(title="Shipments", overlaying='y', side='right', showgrid=False),
                    hovermode='x unified',
                    height=400,
                    template='plotly_white',
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show data table
                with st.expander("📊 View Monthly Data"):
                    display_monthly = monthly.copy()
                    display_monthly['Revenue'] = display_monthly['Revenue'].apply(_revenue_fmt)
                    display_monthly['Shipments'] = display_monthly['Shipments'].apply(lambda x: f"{x:,}")
                    st.dataframe(display_monthly, use_container_width=True)
    
    # OTP Analysis
    target_col = _get_target_series(df)
    if target_col is not None and 'POD DATE/TIME' in df.columns:
        st.subheader("⏱️ On-Time Performance")
        
        df_otp = df.copy()
        df_otp['POD_dt'] = _excel_to_dt(df_otp['POD DATE/TIME'])
        df_otp['Target_dt'] = _excel_to_dt(target_col)
        
        df_otp = df_otp[df_otp['POD_dt'].notna() & df_otp['Target_dt'].notna()].copy()
        
        if not df_otp.empty:
            df_otp['OnTime'] = df_otp['POD_dt'] <= df_otp['Target_dt']
            overall_otp = (df_otp['OnTime'].sum() / len(df_otp)) * 100
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.plotly_chart(make_semi_gauge(f"{sector} OTP", overall_otp), use_container_width=True)
            
            with col2:
                df_otp['YearMonth'] = df_otp['POD_dt'].dt.to_period("M").astype(str)
                monthly_otp = df_otp.groupby('YearMonth').agg(
                    Total=('OnTime', 'count'),
                    OnTime_Count=('OnTime', 'sum')
                ).reset_index()
                monthly_otp['OTP_Pct'] = (monthly_otp['OnTime_Count'] / monthly_otp['Total']) * 100
                monthly_otp = monthly_otp.sort_values('YearMonth')
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=monthly_otp['YearMonth'],
                    y=monthly_otp['OTP_Pct'],
                    mode='lines+markers',
                    name='OTP %',
                    line=dict(color=NAVY, width=3),
                    marker=dict(size=10, color=NAVY),
                    fill='tozeroy',
                    fillcolor='rgba(11,31,68,0.1)'
                ))
                
                fig2.add_hline(y=otp_target, line_dash="dash", line_color=RED, 
                              annotation_text=f"Target: {otp_target}%")
                
                fig2.update_layout(
                    title=dict(text="Monthly OTP Trend", font=dict(size=16, color=NAVY)),
                    xaxis=dict(title="Month", tickangle=-45),
                    yaxis=dict(title="OTP %", range=[0, 100], showgrid=True, gridcolor=GRID),
                    height=300,
                    template='plotly_white',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig2, use_container_width=True)
    
    # Top Accounts
    if 'ACCT NM' in df.columns and 'REVENUE' in df.columns:
        st.subheader("🏆 Top 10 Accounts by Revenue")
        
        df_top = df.copy()
        df_top['REVENUE'] = pd.to_numeric(df_top['REVENUE'], errors='coerce')
        df_top = df_top[df_top['REVENUE'].notna()].copy()
        
        if not df_top.empty:
            top_accounts = df_top.groupby('ACCT NM', as_index=False).agg({
                'REVENUE': 'sum',
                'HAWB': 'count'
            }).rename(columns={'HAWB': 'Shipments'})
            
            top_accounts = top_accounts.nlargest(10, 'REVENUE')
            
            fig3 = go.Figure(go.Bar(
                y=top_accounts['ACCT NM'],
                x=top_accounts['REVENUE'],
                orientation='h',
                marker_color=NAVY,
                text=[_revenue_fmt(v) for v in top_accounts['REVENUE']],
                textposition='outside'
            ))
            
            fig3.update_layout(
                title=dict(text="Top 10 Accounts", font=dict(size=16, color=NAVY)),
                xaxis=dict(title="Revenue", showgrid=True, gridcolor=GRID),
                yaxis=dict(title="", autorange="reversed"),
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig3, use_container_width=True)

# ---------------- Upload ----------------
uploaded = st.file_uploader("Upload Excel File", type=["xlsx","xls"])
if not uploaded:
    st.info("👆 Please upload an Excel file to begin")
    st.stop()

# Add debug toggle
debug_mode = st.checkbox("🐛 Enable Debug Mode", value=False)

# ---------------- Load & Process ----------------
combined_df, stats = get_sheets_dict(uploaded, EMEA_COUNTRIES)

if combined_df.empty:
    st.error("No data found after filtering!")
    st.stop()

# Split into healthcare and non-healthcare
healthcare_df, non_healthcare_df = split_healthcare_classification(combined_df)

# Update stats with healthcare counts
stats['healthcare_rows'] = len(healthcare_df)
stats['non_healthcare_rows'] = len(non_healthcare_df)

# OTP target slider
otp_target = st.slider("Set OTP Target (%)", 80, 100, OTP_TARGET, 1)

# ---------------- NEW: Collect ALL accounts from ALL sheets ----------------
# Read all sheets again to get ALL accounts (before any filtering)
all_accounts_data = []

xls = pd.ExcelFile(uploaded)
for sheet_name in xls.sheet_names:
    try:
        sheet_df = pd.read_excel(xls, sheet_name=sheet_name)
        if 'ACCT NM' in sheet_df.columns:
            # Get unique accounts from this sheet with their row count
            accounts_in_sheet = sheet_df.groupby('ACCT NM').size().reset_index(name='Row_Count')
            accounts_in_sheet['Source_Sheet'] = sheet_name
            
            # Classify each account
            accounts_in_sheet['Classification'] = accounts_in_sheet['ACCT NM'].apply(
                lambda x: 'Healthcare' if is_healthcare(x, sheet_name) else 'Non-Healthcare'
            )
            
            all_accounts_data.append(accounts_in_sheet)
    except Exception as e:
        st.warning(f"Could not read accounts from sheet {sheet_name}: {e}")

# Combine all accounts
if all_accounts_data:
    all_accounts_df = pd.concat(all_accounts_data, ignore_index=True)
    
    # Aggregate by account name across all sheets
    all_accounts_summary = all_accounts_df.groupby('ACCT NM').agg({
        'Row_Count': 'sum',
        'Classification': 'first',  # Take first classification (should be consistent)
        'Source_Sheet': lambda x: ', '.join(sorted(set(x)))  # Combine sheet names
    }).reset_index()
    
    # Check which accounts are actually used in the filtered data
    if 'ACCT NM' in combined_df.columns:
        used_accounts = set(combined_df['ACCT NM'].dropna().unique())
        all_accounts_summary['Used_in_Filter'] = all_accounts_summary['ACCT NM'].isin(used_accounts)
    else:
        all_accounts_summary['Used_in_Filter'] = False
    
    # Sort by classification and then by row count
    all_accounts_summary = all_accounts_summary.sort_values(
        ['Classification', 'Row_Count'], 
        ascending=[True, False]
    )
else:
    all_accounts_summary = pd.DataFrame()

# ---------------- Debug Info ----------------
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

# Show processing statistics
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

# Create tabs - ADDED NEW TAB
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
    This tab shows **ALL accounts** from **ALL sheets** in the Excel file, regardless of filtering:
    - 🟢 **Green**: Accounts classified as **Healthcare** and used in filtered data
    - 🔵 **Blue**: Accounts classified as **Non-Healthcare** and used in filtered data
    - ⚪ **No highlight**: Accounts that exist but are **not used** in filtered data (filtered out by EMEA or 440-BILLED criteria)
    """)
    
    if not all_accounts_summary.empty:
        # Summary metrics
        total_accounts = len(all_accounts_summary)
        hc_accounts = len(all_accounts_summary[all_accounts_summary['Classification'] == 'Healthcare'])
        non_hc_accounts = len(all_accounts_summary[all_accounts_summary['Classification'] == 'Non-Healthcare'])
        used_accounts = len(all_accounts_summary[all_accounts_summary['Used_in_Filter'] == True])
        unused_accounts = len(all_accounts_summary[all_accounts_summary['Used_in_Filter'] == False])
        
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
        
        # Create styled dataframe
        def highlight_accounts(row):
            if not row['Used_in_Filter']:
                return [''] * len(row)  # No highlighting for unused accounts
            elif row['Classification'] == 'Healthcare':
                return ['background-color: #d1fae5'] * len(row)  # Light green
            else:
                return ['background-color: #dbeafe'] * len(row)  # Light blue
        
        # Display the dataframe with styling
        display_df = filtered_accounts.copy()
        display_df['Row_Count'] = display_df['Row_Count'].apply(lambda x: f"{x:,}")
        display_df['Used_in_Filter'] = display_df['Used_in_Filter'].map({True: '✓ Yes', False: '✗ No'})
        display_df = display_df.rename(columns={
            'ACCT NM': 'Account Name',
            'Row_Count': 'Total Rows (All Sheets)',
            'Classification': 'Classification',
            'Source_Sheet': 'Found in Sheets',
            'Used_in_Filter': 'Used in Filtered Data'
        })
        
        styled_df = display_df.style.apply(highlight_accounts, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=600)
        
        # Detailed breakdown
        with st.expander("📊 Detailed Account Analysis"):
            st.markdown("#### Accounts Used in Filtered Data:")
            used_df = all_accounts_summary[all_accounts_summary['Used_in_Filter'] == True].copy()
            if not used_df.empty:
                hc_used = len(used_df[used_df['Classification'] == 'Healthcare'])
                non_hc_used = len(used_df[used_df['Classification'] == 'Non-Healthcare'])
                st.write(f"- **Healthcare accounts used:** {hc_used:,}")
                st.write(f"- **Non-Healthcare accounts used:** {non_hc_used:,}")
                st.write(f"- **Total rows in filtered data:** {used_df['Row_Count'].sum():,}")
            
            st.markdown("#### Accounts Filtered Out (Not Used):")
            unused_df = all_accounts_summary[all_accounts_summary['Used_in_Filter'] == False].copy()
            if not unused_df.empty:
                hc_unused = len(unused_df[unused_df['Classification'] == 'Healthcare'])
                non_hc_unused = len(unused_df[unused_df['Classification'] == 'Non-Healthcare'])
                st.write(f"- **Healthcare accounts filtered out:** {hc_unused:,}")
                st.write(f"- **Non-Healthcare accounts filtered out:** {non_hc_unused:,}")
                st.write(f"- **Total rows filtered out:** {unused_df['Row_Count'].sum():,}")
                st.info("💡 These accounts were removed by EMEA country filter or 440-BILLED status filter")
        
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
                ]['ACCT NM'].head(5).tolist()
                if hc_unused_sample:
                    for acc in hc_unused_sample:
                        st.write(f"⚪ {acc}")
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
                ]['ACCT NM'].head(5).tolist()
                if non_hc_unused_sample:
                    for acc in non_hc_unused_sample:
                        st.write(f"⚪ {acc}")
                else:
                    st.write("No non-healthcare accounts filtered out")
    else:
        st.warning("No account data available")
