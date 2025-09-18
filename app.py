import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

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

    # Row-level OTP calculation
    ok = d["_pod"].notna() & d["_target"].notna()
    d["On_Time_Gross"] = False
    d.loc[ok, "On_Time_Gross"] = d.loc[ok, "_pod"] <= d.loc[ok, "_target"]
    d["Late"] = ~d["On_Time_Gross"]
    d["On_Time_Net"] = d["On_Time_Gross"] | (d["Late"] & ~d["Is_Controllable"])

    return d

@st.cache_data(show_spinner=False)
def monthly_frames(d: pd.DataFrame):
    """Build Monthly OTP, Volume, Pieces — all by POD month with the same keys."""
    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Filter to only rows with valid POD dates
    base_vol = d.dropna(subset=["_pod"]).copy()
    
    if base_vol.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Group by POD month for volume
    vol = (base_vol.groupby(["Month_YYYY_MM","Month_Display","Month_Sort"], as_index=False)
                 .size().rename(columns={"size":"Volume"}))

    # Group by POD month for pieces
    pieces = (base_vol.groupby(["Month_YYYY_MM","Month_Display","Month_Sort"], as_index=False)
                      .agg(Pieces=("PIECES","sum")))

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
    vol, pieces, otp = [x.sort_values("Month_Sort") for x in (vol, pieces, otp)]
    return vol, pieces, otp

def calc_summary(d: pd.DataFrame):
    """Calculate summary statistics."""
    if d.empty:
        return np.nan, np.nan, 0, 0, 0, 0
    
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
    return (round(gross,2) if pd.notna(gross) else np.nan,
            round(net,2)   if pd.notna(net)   else np.nan,
            volume_total, exceptions, controllables, uncontrollables)

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
    
    vol_pod, pieces_pod, otp_pod = monthly_frames(processed_df)
    gross_otp, net_otp, volume_total, exceptions, controllables, uncontrollables = calc_summary(processed_df)
    
    # ---------------- KPIs & Gauges ----------------
    left, right = st.columns([1, 1.5])
    with left:
        st.markdown(f'<div class="kpi"><div class="k-num">{volume_total:,}</div><div class="k-cap">Volume (rows with POD)</div></div>', unsafe_allow_html=True)
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
    - **Gross OTP** = `POD ≤ target (UPD DEL → QDT)`.
    - **Net/Adjusted OTP** = counts **non-controllable** lates as on-time (QC NAME contains Agent/Delivery agent/Customs/Warehouse) ⇒ **Net ≥ Gross**.
    """)

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
        st.plotly_chart(fig, use_container_width=True)
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

        # ---------------- Chart: 5 Worst Accounts by Net OTP — August (exclude 0% & NaN) ----------------
    st.subheader(f"{tab_name}: 5 Worst Accounts by Net OTP — August")
    if 'ACCT NM' in processed_df.columns:
        base = processed_df.dropna(subset=['_pod', '_target']).copy()
        if not base.empty:
            base['Month'] = base['_pod'].dt.month
            base['Year']  = base['_pod'].dt.year
            aug_df = base[base['Month'] == 8]

            if not aug_df.empty:
                latest_aug_year = int(aug_df['Year'].max())
                aug_latest = aug_df[aug_df['Year'] == latest_aug_year].copy()

                grp = (aug_latest.groupby('ACCT NM', dropna=False)
                                .agg(Net_On=('On_Time_Net', 'sum'),
                                     Tot=('On_Time_Net', 'count'))
                                .reset_index())
                # Compute Net OTP and filter out NaN and 0%
                grp['Net_OTP'] = (grp['Net_On'] / grp['Tot'] * 100)
                grp = grp[grp['Net_OTP'].notna() & (grp['Net_OTP'] > 0)]

                if not grp.empty:
                    worst = grp.nsmallest(5, 'Net_OTP').copy()
                    worst['Net_OTP'] = worst['Net_OTP'].round(2)

                    figw = go.Figure()
                    figw.add_trace(go.Bar(
                        x=worst['Net_OTP'],
                        y=worst['ACCT NM'].astype(str),
                        orientation='h',
                        marker_color=NAVY,
                        text=[f"{v:.2f}%" for v in worst['Net_OTP']],
                        textposition='outside'
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
                        height=360,
                        plot_bgcolor="white",
                        margin=dict(l=10, r=40, t=20, b=40),
                        xaxis=dict(title="Net OTP (%)", range=[0, 110], gridcolor=GRID, showgrid=True),
                        yaxis=dict(title="", automargin=True)
                    )
                    st.plotly_chart(figw, use_container_width=True)
                else:
                    st.info(f"No non-null, >0% Net OTP accounts for August {latest_aug_year}.")
            else:
                st.info("No August data available in the current dataset.")
        else:
            st.info("No rows with both POD and target available for account-level OTP in August.")
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
    
    **Data Scope:**
    - EMEA countries only
    - STATUS = 440-BILLED
    - Month grouping by POD DATE/TIME
    """)

if not uploaded_file:
    st.info("""
    👆 **Please upload your Excel file to begin.**
    
    **Expected file structure:**
    - Multiple sheets (Aviation SVC, MNX Charter, AMS, LDN, Americas International Desk)
    - Required columns: PU CTRY, STATUS, POD DATE/TIME, ACCT NM
    - Optional columns: UPD DEL, QDT, QC NAME, PIECES
    
    **Data processing:**
    - Filters for EMEA countries only
    - Filters for STATUS = 440-BILLED
    - Categorizes accounts as Healthcare or Non-Healthcare
    - Calculates OTP metrics by POD month
    """)
    st.stop()

# Process uploaded file
with st.spinner("Processing Excel file..."):
    healthcare_df, non_healthcare_df, stats = read_and_combine_sheets(uploaded_file)

# Detailed validation info
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
        
        # Additional validation
        st.markdown("#### Validation:")
        total_processed = stats.get('healthcare_rows', 0) + stats.get('non_healthcare_rows', 0)
        st.write(f"✓ Total after filters: {stats.get('status_filtered', 0)}")
        st.write(f"✓ HC + Non-HC total: {total_processed}")
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
