#Code 2 - Enhanced Version with All Accounts Overview Tab
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
.healthcare-row { background-color: #d4f4dd !important; }
.non-healthcare-row { background-color: #d6e5ff !important; }
.unused-row { background-color: #f9f9f9 !important; }
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
    
    # Note: Removed the exclusion list to ensure proper categorization
    # EXCLUDE_FROM_HEALTHCARE = {"avid", "lantheus", "life"}  
    # lower = str(account_name).strip().lower()
    # if any(excluded in lower for excluded in EXCLUDE_FROM_HEALTHCARE):
    #     return False
      
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
    """Semi-circular gauge chart."""
    if value >= 80:
        color = "#10b981"
    elif value >= 60:
        color = "#f59e0b"
    else:
        color = "#ef4444"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 18, "color": SLATE}},
        number={"suffix": "%", "font": {"size": 36, "color": NAVY}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": GRID},
            "bar": {"color": color},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": GRID,
            "steps": [
                {"range": [0, 60], "color": "#fee2e2"},
                {"range": [60, 80], "color": "#fed7aa"},
                {"range": [80, 100], "color": "#d1fae5"}
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": OTP_TARGET
            }
        }
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif"}
    )
    return fig

def create_dashboard_view(df: pd.DataFrame, sector: str, otp_target: float, debug_mode: bool = False):
    """Create the dashboard view for a sector."""
    if df.empty:
        st.info(f"No {sector} data available after filtering")
        return
    
    # Show column information if in debug mode
    if debug_mode:
        with st.expander(f"🔍 Debug: {sector} DataFrame Info"):
            st.write(f"Shape: {df.shape}")
            st.write("Columns:", list(df.columns))
            st.write("First few rows:")
            st.dataframe(df.head())
    
    # Get POD and Target dates
    pod_col = "POD DATE/TIME" if "POD DATE/TIME" in df.columns else None
    if not pod_col:
        st.warning(f"No POD DATE/TIME column found in {sector} data")
        return
    
    df["POD_DATE"] = _excel_to_dt(df[pod_col])
    target_series = _get_target_series(df)
    
    if target_series is not None:
        df["TARGET_DATE"] = _excel_to_dt(target_series)
        df["ON_TIME"] = df["POD_DATE"] <= df["TARGET_DATE"]
        has_target = True
    else:
        has_target = False
    
    # Calculate metrics
    total_shipments = len(df)
    
    if has_target:
        valid_otp = df[df["TARGET_DATE"].notna() & df["POD_DATE"].notna()]
        otp_rate = (valid_otp["ON_TIME"].sum() / len(valid_otp) * 100) if len(valid_otp) > 0 else 0
        on_time_shipments = valid_otp["ON_TIME"].sum()
    else:
        otp_rate = 0
        on_time_shipments = 0
    
    # Calculate month-over-month data
    df["Month"] = df["POD_DATE"].dt.to_period("M").astype(str)
    valid_months = df[df["Month"].notna() & (df["Month"] != "NaT")]
    
    if not valid_months.empty:
        month_counts = valid_months.groupby("Month").size()
        
        if has_target:
            month_otp = valid_months[valid_months["TARGET_DATE"].notna()].groupby("Month").apply(
                lambda x: (x["ON_TIME"].sum() / len(x) * 100) if len(x) > 0 else 0
            ).round(1)
        else:
            month_otp = pd.Series(dtype=float)
    else:
        month_counts = pd.Series(dtype=int)
        month_otp = pd.Series(dtype=float)
    
    # Display KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="kpi">
            <div class="k-num">{total_shipments:,}</div>
            <div class="k-cap">Total Shipments</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi">
            <div class="k-num" style="color: {'#10b981' if otp_rate >= otp_target else '#ef4444'}">
                {otp_rate:.1f}%
            </div>
            <div class="k-cap">On-Time Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="kpi">
            <div class="k-num">{on_time_shipments:,}</div>
            <div class="k-cap">On-Time Deliveries</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        unique_accounts = df['ACCT NM'].nunique() if 'ACCT NM' in df.columns else 0
        st.markdown(f"""
        <div class="kpi">
            <div class="k-num">{unique_accounts}</div>
            <div class="k-cap">Unique Accounts</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts section
    st.markdown("### 📊 Performance Trends")
    
    if not month_counts.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Volume trend
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=month_counts.index,
                y=month_counts.values,
                name="Shipments",
                marker_color=NAVY,
                text=month_counts.values,
                textposition='outside'
            ))
            fig.update_layout(
                title="Monthly Shipment Volume",
                xaxis_title="Month",
                yaxis_title="Number of Shipments",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # OTP trend (if available)
            if has_target and not month_otp.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=month_otp.index,
                    y=month_otp.values,
                    mode='lines+markers',
                    name="OTP %",
                    line=dict(color=BLUE, width=3),
                    marker=dict(size=8)
                ))
                fig.add_hline(y=otp_target, line_dash="dash", line_color="red",
                            annotation_text=f"Target: {otp_target}%")
                fig.update_layout(
                    title="Monthly On-Time Performance",
                    xaxis_title="Month",
                    yaxis_title="On-Time %",
                    yaxis_range=[0, 105],
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.plotly_chart(make_semi_gauge("On-Time Performance", otp_rate), use_container_width=True)
    
    # Top accounts analysis
    if 'ACCT NM' in df.columns:
        st.markdown("### 🏆 Top Accounts")
        account_stats = df.groupby('ACCT NM').agg({
            pod_col: 'count',
            'ON_TIME': lambda x: (x.sum() / len(x) * 100) if has_target else 0
        }).round(1)
        account_stats.columns = ['Shipments', 'OTP %']
        account_stats = account_stats.sort_values('Shipments', ascending=False).head(10)
        
        # Create bar chart for top accounts
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=account_stats.index,
            y=account_stats['Shipments'],
            name="Shipments",
            marker_color=NAVY,
            yaxis='y',
            text=account_stats['Shipments'],
            textposition='outside'
        ))
        
        if has_target:
            fig.add_trace(go.Scatter(
                x=account_stats.index,
                y=account_stats['OTP %'],
                name="OTP %",
                yaxis='y2',
                mode='lines+markers',
                marker=dict(color=GOLD, size=10),
                line=dict(color=GOLD, width=3)
            ))
        
        fig.update_layout(
            title="Top 10 Accounts by Volume",
            xaxis_tickangle=-45,
            height=400,
            yaxis=dict(title="Shipments", side='left'),
            yaxis2=dict(title="OTP %", side='right', overlaying='y', range=[0, 105]) if has_target else None,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed table
        with st.expander("📋 Detailed Account Statistics"):
            st.dataframe(account_stats.style.format({
                'Shipments': '{:,.0f}',
                'OTP %': '{:.1f}%'
            }))

def create_accounts_overview(all_df: pd.DataFrame, healthcare_df: pd.DataFrame, 
                            non_healthcare_df: pd.DataFrame, stats: dict):
    """Create the All Accounts Overview tab showing all accounts with color coding."""
    
    st.markdown("## 📋 All Accounts Overview")
    
    # Get all unique accounts from the raw data
    all_accounts = set()
    healthcare_accounts = set()
    non_healthcare_accounts = set()
    
    if 'ACCT NM' in all_df.columns:
        all_accounts = set(all_df['ACCT NM'].dropna().unique())
    
    if not healthcare_df.empty and 'ACCT NM' in healthcare_df.columns:
        healthcare_accounts = set(healthcare_df['ACCT NM'].dropna().unique())
    
    if not non_healthcare_df.empty and 'ACCT NM' in non_healthcare_df.columns:
        non_healthcare_accounts = set(non_healthcare_df['ACCT NM'].dropna().unique())
    
    # Find unused accounts (in raw data but not in filtered data)
    used_accounts = healthcare_accounts | non_healthcare_accounts
    unused_accounts = all_accounts - used_accounts
    
    # Display summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Unique Accounts", len(all_accounts))
    with col2:
        st.metric("Healthcare Accounts", len(healthcare_accounts))
    with col3:
        st.metric("Non-Healthcare Accounts", len(non_healthcare_accounts))
    with col4:
        st.metric("Unused Accounts", len(unused_accounts))
    
    # Create account categorization DataFrame
    accounts_data = []
    
    # Process each account
    for account in all_accounts:
        category = "Unused"
        color = "#f9f9f9"  # Light gray for unused
        
        if account in healthcare_accounts:
            category = "Healthcare"
            color = "#d4f4dd"  # Light green
        elif account in non_healthcare_accounts:
            category = "Non-Healthcare"
            color = "#d6e5ff"  # Light blue
        
        # Get shipment count for this account
        shipment_count = 0
        if 'ACCT NM' in all_df.columns:
            shipment_count = len(all_df[all_df['ACCT NM'] == account])
        
        # Get source sheets for this account
        source_sheets = []
        if 'ACCT NM' in all_df.columns and 'Source_Sheet' in all_df.columns:
            source_sheets = all_df[all_df['ACCT NM'] == account]['Source_Sheet'].unique().tolist()
        
        accounts_data.append({
            'Account Name': account,
            'Category': category,
            'Total Shipments': shipment_count,
            'Source Sheets': ', '.join(source_sheets) if source_sheets else 'N/A',
            '_color': color
        })
    
    # Create DataFrame and sort by category and shipment count
    accounts_df = pd.DataFrame(accounts_data)
    accounts_df = accounts_df.sort_values(['Category', 'Total Shipments'], 
                                          ascending=[True, False])
    
    # Display color legend
    st.markdown("""
    ### Color Legend:
    - 🟢 **Green Background**: Healthcare Accounts (filtered and included in Healthcare analysis)
    - 🔵 **Blue Background**: Non-Healthcare Accounts (filtered and included in Non-Healthcare analysis)
    - ⚪ **Gray Background**: Unused Accounts (not meeting filter criteria or excluded from analysis)
    """)
    
    # Add filters
    st.markdown("### 🔍 Filter Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        category_filter = st.multiselect(
            "Filter by Category",
            options=['Healthcare', 'Non-Healthcare', 'Unused'],
            default=['Healthcare', 'Non-Healthcare', 'Unused']
        )
    
    with col2:
        min_shipments = st.number_input(
            "Minimum Shipments",
            min_value=0,
            max_value=int(accounts_df['Total Shipments'].max()) if not accounts_df.empty else 0,
            value=0
        )
    
    with col3:
        search_term = st.text_input("Search Account Name", "")
    
    # Apply filters
    filtered_df = accounts_df[accounts_df['Category'].isin(category_filter)]
    filtered_df = filtered_df[filtered_df['Total Shipments'] >= min_shipments]
    
    if search_term:
        filtered_df = filtered_df[
            filtered_df['Account Name'].str.contains(search_term, case=False, na=False)
        ]
    
    # Display the table with color coding
    st.markdown(f"### 📊 Accounts Table (Showing {len(filtered_df)} of {len(accounts_df)} accounts)")
    
    # Style the dataframe
    def highlight_rows(row):
        if row['Category'] == 'Healthcare':
            return ['background-color: #d4f4dd'] * len(row)
        elif row['Category'] == 'Non-Healthcare':
            return ['background-color: #d6e5ff'] * len(row)
        else:
            return ['background-color: #f9f9f9'] * len(row)
    
    styled_df = filtered_df[['Account Name', 'Category', 'Total Shipments', 'Source Sheets']].style.apply(
        highlight_rows, axis=1
    ).format({'Total Shipments': '{:,.0f}'})
    
    st.dataframe(styled_df, height=600, use_container_width=True)
    
    # Export functionality
    st.markdown("### 💾 Export Data")
    csv = filtered_df[['Account Name', 'Category', 'Total Shipments', 'Source Sheets']].to_csv(index=False)
    st.download_button(
        label="Download Accounts Data as CSV",
        data=csv,
        file_name=f"accounts_overview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Additional Analysis
    with st.expander("📈 Account Distribution Analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart of account categories
            fig = px.pie(
                values=[len(healthcare_accounts), len(non_healthcare_accounts), len(unused_accounts)],
                names=['Healthcare', 'Non-Healthcare', 'Unused'],
                title="Account Distribution by Category",
                color_discrete_map={
                    'Healthcare': '#10b981',
                    'Non-Healthcare': '#3b82f6',
                    'Unused': '#9ca3af'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bar chart of top sheets by account count
            if 'Source_Sheet' in all_df.columns and 'ACCT NM' in all_df.columns:
                sheet_accounts = all_df.groupby('Source_Sheet')['ACCT NM'].nunique().sort_values(ascending=False)
                fig = go.Figure(go.Bar(
                    x=sheet_accounts.values,
                    y=sheet_accounts.index,
                    orientation='h',
                    marker_color=NAVY
                ))
                fig.update_layout(
                    title="Unique Accounts per Sheet",
                    xaxis_title="Number of Accounts",
                    yaxis_title="Sheet Name",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

# ---------------- Main App ----------------
uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx', 'xls'], key="file_uploader")

debug_mode = st.sidebar.checkbox("Debug Mode", value=False)
otp_target = st.sidebar.slider("OTP Target (%)", 0, 100, OTP_TARGET, 5)

if uploaded_file:
    with st.spinner("Processing file..."):
        # Read all sheets
        all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
        
        combined_dfs = []
        all_raw_dfs = []  # Store all raw data for account overview
        stats = {
            'sheets_read': list(all_sheets.keys()),
            'total_rows_raw': 0,
            'total_combined_raw': 0,
            'by_sheet': {},
            'emea_rows': 0,
            'status_filtered': 0,
            'healthcare_rows': 0,
            'non_healthcare_rows': 0,
            'all_accounts': set(),
            'healthcare_accounts': set(),
            'non_healthcare_accounts': set()
        }
        
        # Process each sheet
        for sheet_name, df_sheet in all_sheets.items():
            try:
                if df_sheet.empty:
                    continue
                
                # Track raw data
                raw_rows = len(df_sheet)
                stats['total_rows_raw'] += raw_rows
                
                # Store raw data with sheet name for account overview
                df_sheet_copy = df_sheet.copy()
                df_sheet_copy['Source_Sheet'] = sheet_name
                all_raw_dfs.append(df_sheet_copy)
                
                # Collect all accounts before filtering
                if 'ACCT NM' in df_sheet.columns:
                    stats['all_accounts'].update(df_sheet['ACCT NM'].dropna().unique())
                
                sheet_stats = {
                    'raw_rows': raw_rows,
                    'initial': len(df_sheet),
                    'emea': 0,
                    'final': 0
                }
                
                # EMEA Filter
                if "ORIGIN" in df_sheet.columns and "DEST" in df_sheet.columns:
                    orig = df_sheet["ORIGIN"].str.upper().str[:2]
                    dest = df_sheet["DEST"].str.upper().str[:2]
                    emea_mask = orig.isin(EMEA_COUNTRIES) | dest.isin(EMEA_COUNTRIES)
                    df_emea = df_sheet[emea_mask].copy()
                else:
                    df_emea = df_sheet.copy()
                
                sheet_stats['emea'] = len(df_emea)
                stats['emea_rows'] += len(df_emea)
                
                # Status Filter (440-BILLED)
                if "STATUS" in df_emea.columns:
                    status_mask = df_emea["STATUS"].astype(str).str.strip().str.upper() == "440-BILLED"
                    df_filtered = df_emea[status_mask].copy()
                else:
                    df_filtered = df_emea.copy()
                
                sheet_stats['final'] = len(df_filtered)
                stats['by_sheet'][sheet_name] = sheet_stats
                
                # Add source sheet column
                df_filtered["Source_Sheet"] = sheet_name
                
                # Classify accounts
                if 'ACCT NM' in df_filtered.columns:
                    df_filtered['Is_Healthcare'] = df_filtered['ACCT NM'].apply(
                        lambda x: is_healthcare(x, sheet_name)
                    )
                    
                    # Track healthcare and non-healthcare accounts
                    hc_accounts = df_filtered[df_filtered['Is_Healthcare'] == True]['ACCT NM'].dropna().unique()
                    non_hc_accounts = df_filtered[df_filtered['Is_Healthcare'] == False]['ACCT NM'].dropna().unique()
                    
                    stats['healthcare_accounts'].update(hc_accounts)
                    stats['non_healthcare_accounts'].update(non_hc_accounts)
                else:
                    # Default classification based on sheet
                    df_filtered['Is_Healthcare'] = is_healthcare(None, sheet_name)
                
                combined_dfs.append(df_filtered)
                
            except Exception as e:
                st.warning(f"Error processing sheet '{sheet_name}': {str(e)}")
                stats['by_sheet'][sheet_name] = {'error': str(e)}
        
        # Combine all sheets
        if combined_dfs:
            combined_df = pd.concat(combined_dfs, ignore_index=True)
            stats['status_filtered'] = len(combined_df)
            
            # Combine all raw data
            all_raw_df = pd.concat(all_raw_dfs, ignore_index=True) if all_raw_dfs else pd.DataFrame()
            stats['total_combined_raw'] = len(all_raw_df)
            
            # Split into healthcare and non-healthcare
            healthcare_df = combined_df[combined_df['Is_Healthcare'] == True].copy()
            non_healthcare_df = combined_df[combined_df['Is_Healthcare'] == False].copy()
            
            stats['healthcare_rows'] = len(healthcare_df)
            stats['non_healthcare_rows'] = len(non_healthcare_df)
        else:
            combined_df = pd.DataFrame()
            all_raw_df = pd.DataFrame()
            healthcare_df = pd.DataFrame()
            non_healthcare_df = pd.DataFrame()
    
    # Debug information
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
        st.write(f"   Total Unique Accounts: {len(stats.get('all_accounts', set())):,}")
        st.write(f"   Healthcare Accounts: {len(stats.get('healthcare_accounts', set())):,}")
        st.write(f"   Non-Healthcare Accounts: {len(stats.get('non_healthcare_accounts', set())):,}")
        
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

    # Create tabs with the new All Accounts tab
    tab1, tab2, tab3 = st.tabs(["📋 All Accounts Overview", "🏥 Healthcare", "✈️ Non-Healthcare"])
    
    with tab1:
        create_accounts_overview(all_raw_df, healthcare_df, non_healthcare_df, stats)
    
    with tab2:
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
    
    with tab3:
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
