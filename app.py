import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import base64
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from streamlit_option_menu import option_menu
import calendar

# Set page configuration
st.set_page_config(
    page_title="SR Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
def set_custom_theme():
    st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stApp {
        color: #1e2a3a;
    }
    .stDataFrame, .stTable {
        background-color: white;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    .status-badge {
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
    }
    .badge-pending {
        background-color: #ffecb3;
        color: #b17825;
    }
    .badge-complete {
        background-color: #c8e6c9;
        color: #2e7d32;
    }
    .badge-in-progress {
        background-color: #bbdefb;
        color: #1565c0;
    }
    .badge-cancelled {
        background-color: #ffcdd2;
        color: #c62828;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
        margin: 0;
    }
    .metric-label {
        font-size: 1em;
        color: #888;
        margin: 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e88e5 !important;
        color: white !important;
    }
    h1, h2, h3 {
        color: #1565c0;
    }
    .stProgress .st-eb {
        background-color: #bbdefb;
    }
    .stProgress .st-ec {
        background-color: #1976d2;
    }
    </style>
    """, unsafe_allow_html=True)

set_custom_theme()

# Initialize session state for caching and storing data
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'main_df' not in st.session_state:
    st.session_state.main_df = None
if 'sr_df' not in st.session_state:
    st.session_state.sr_df = None
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = None
if 'last_upload_time' not in st.session_state:
    st.session_state.last_upload_time = None
if 'selected_users' not in st.session_state:
    st.session_state.selected_users = []

# Function to load and process data
@st.cache_data
def load_data(file):
    return pd.read_excel(file)

# Function to process main dataframe
def process_main_df(df):
    # Ensure date columns are in datetime format
    date_columns = ['Case Start Date', 'Last Updated']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Extract all unique users
    if 'Current User Id' in df.columns:
        all_users = sorted(df['Current User Id'].dropna().unique().tolist())
        st.session_state.all_users = all_users
    
    return df

# Function to classify and extract ticket info
def classify_and_extract(note):
    if not isinstance(note, str):
        return "Not Triaged", None, None
    
    note_lower = note.lower()
    # Enhanced regex pattern to catch more variations
    sr_pattern = r'(tkt|sr|inc|ticket|مرجعي|incident|اس ار|انسدنت|service request|#|reference|ref)[\s\W]{0,50}?(\d{4,})'
    match = re.search(sr_pattern, note_lower)
    
    if match:
        ticket_num = int(match.group(2))
        # SR numbers typically between 14000-16000 (adjust based on your system)
        ticket_type = "SR" if 14000 <= ticket_num <= 16000 else "Incident"
        return "Pending SR/Incident", ticket_num, ticket_type
    
    return "Not Triaged", None, None

# Function to calculate case age in days
def calculate_age(start_date):
    if pd.isna(start_date):
        return None
    return (datetime.now() - start_date).days

# Function to calculate SLA status
def calculate_sla_status(age_days, sr_status=None):
    if pd.isna(age_days):
        return "Unknown", 0
    
    # SLA thresholds - customize based on your requirements
    if sr_status in ["Completed", "Cancelled", "Closed"]:
        return "Completed", 100
    elif age_days <= 3:
        return "Within SLA", min(100, 100 - (age_days / 3 * 100))
    elif age_days <= 5:
        return "At Risk", max(0, 100 - (age_days / 5 * 150))
    else:
        return "Breached", 0

# Function to generate color for SLA status
def get_sla_color(status):
    colors = {
        "Within SLA": "#2e7d32",  # Green
        "At Risk": "#ff9800",     # Orange
        "Breached": "#c62828",    # Red
        "Completed": "#1565c0",   # Blue
        "Unknown": "#9e9e9e"      # Grey
    }
    return colors.get(status, "#9e9e9e")

# Function to create downloadable Excel
def generate_excel_download(data):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        data.to_excel(writer, index=False, sheet_name='Results')
        workbook = writer.book
        worksheet = writer.sheets['Results']
        
        # Add formats for better Excel styling
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#1976d2',
            'color': 'white',
            'border': 1
        })
        
        # Apply header format
        for col_num, value in enumerate(data.columns.values):
            worksheet.write(0, col_num, value, header_format)
            
        # Auto-adjust columns' width
        for i, col in enumerate(data.columns):
            max_len = max(data[col].astype(str).apply(len).max(), len(str(col))) + 1
            worksheet.set_column(i, i, max_len)
    
    output.seek(0)
    return output

# Function to create Streamlit metrics card
def metric_card(title, value, delta=None, delta_label=None, icon=None):
    st.markdown(f"""
    <div class="card">
        <p class="metric-label">{title}</p>
        <p class="metric-value">{value}</p>
        {f'<p style="color: {"green" if delta >= 0 else "red"};">{delta_label}: {delta}%</p>' if delta is not None else ''}
    </div>
    """, unsafe_allow_html=True)

# Sidebar - File Upload Section
with st.sidebar:
    st.title("📊 SR Analyzer Pro")
    st.markdown("---")

    st.subheader("📁 Data Import")
    uploaded_file = st.file_uploader("Upload Main Excel File (.xlsx)", type="xlsx")
    sr_status_file = st.file_uploader("Upload SR Status Excel (optional)", type="xlsx")
    
    if uploaded_file:
        with st.spinner("Loading main data..."):
            df = load_data(uploaded_file)
            st.session_state.main_df = process_main_df(df)
            st.session_state.last_upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        st.success(f"Main data loaded: {df.shape[0]} records")
        st.session_state.data_loaded = True
    
    if sr_status_file:
        with st.spinner("Loading SR status data..."):
            sr_df = load_data(sr_status_file)
            st.session_state.sr_df = sr_df
        st.success(f"SR status data loaded: {sr_df.shape[0]} records")
    
    if st.session_state.last_upload_time:
        st.info(f"Last upload: {st.session_state.last_upload_time}")
    
    st.markdown("---")

# Main content
if not st.session_state.data_loaded:
    st.title("📊 SR Analyzer Pro")
    st.markdown("""
    ### Welcome to the SR Analyzer Pro!
    
    This application helps you analyze Service Requests and Incidents efficiently.
    
    To get started:
    1. Upload your main Excel file using the sidebar
    2. Optionally upload SR status file for enhanced analysis
    3. Use the dashboard to filter, analyze and export your data
    
    **Features:**
    - Advanced filtering and search
    - Visual analytics and charts
    - SLA monitoring and aging analysis
    - Team performance metrics
    - Export capabilities
    """)
    
    # Sample UI image could be added here
    st.image("https://via.placeholder.com/800x400.png?text=SR+Analyzer+Pro+Dashboard+Preview", use_column_width=True)
else:
    # Process and filter data
    df_main = st.session_state.main_df.copy()
    
    # Prepare tab interface
    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "SR Analysis", "Team Performance", "Settings"],
        icons=["speedometer2", "kanban", "people", "gear"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "margin": "0!important"},
            "icon": {"color": "#1565c0", "font-size": "14px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "center",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#1976d2", "color": "white"},
        }
    )
    
    # Function to further process and enrich data
    def enrich_data(df):
        # Create a copy to avoid modifying the original
        df_enriched = df.copy()
        
        # Classify and extract ticket info
        df_enriched[['Status', 'Ticket Number', 'Type']] = pd.DataFrame(
            df_enriched['Last Note'].apply(lambda x: pd.Series(classify_and_extract(x)))
        )
        
        # Calculate case age
        if 'Case Start Date' in df_enriched.columns:
            df_enriched['Age (Days)'] = df_enriched['Case Start Date'].apply(calculate_age)
        
        # Merge with SR status data if available
        if st.session_state.sr_df is not None:
            sr_df = st.session_state.sr_df.copy()
            
            # Clean and prepare SR data
            sr_df['Service Request'] = sr_df['Service Request'].astype(str).str.extract(r'(\d{4,})')
            sr_df['Service Request'] = pd.to_numeric(sr_df['Service Request'], errors='coerce')
            
            # Rename columns for clarity
            sr_df = sr_df.rename(columns={
                'Status': 'SR Status',
                'LastModDateTime': 'Last Update'
            })
            
            # Merge data
            df_enriched['Ticket Number'] = pd.to_numeric(df_enriched['Ticket Number'], errors='coerce')
            df_enriched = df_enriched.merge(
                sr_df[['Service Request', 'SR Status', 'Last Update']],
                how='left',
                left_on='Ticket Number',
                right_on='Service Request'
            ).drop(columns=['Service Request'])
            
            # Calculate SLA status
            df_enriched[['SLA Status', 'SLA Percent']] = df_enriched.apply(
                lambda row: pd.Series(calculate_sla_status(row['Age (Days)'], row.get('SR Status'))),
                axis=1
            )
        else:
            # Calculate basic SLA without SR status
            df_enriched[['SLA Status', 'SLA Percent']] = df_enriched.apply(
                lambda row: pd.Series(calculate_sla_status(row['Age (Days)'])),
                axis=1
            )
        
        return df_enriched
    
    # Apply user filters and enrichment
    with st.sidebar:
        st.subheader("🔍 Filters")
        
        # Get all users
        all_users = df_main['Current User Id'].dropna().unique().tolist()
        
        # Multi-select for users
        default_users = ['ali.babiker', 'anas.hasan', 'ahmed.mostafa']
        default_users = [u for u in default_users if u in all_users]  # Ensure defaults exist
        
        selected_users = st.multiselect(
            "Select Users", 
            options=all_users,
            default=default_users
        )
        st.session_state.selected_users = selected_users
        
        # Apply filters
        if selected_users:
            df_filtered = df_main[df_main['Current User Id'].isin(selected_users)].copy()
        else:
            df_filtered = df_main.copy()
        
        # Date range filter
        if 'Case Start Date' in df_filtered.columns:
            min_date = df_filtered['Case Start Date'].min().date()
            max_date = df_filtered['Case Start Date'].max().date()
            
            date_range = st.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                df_filtered = df_filtered[
                    (df_filtered['Case Start Date'].dt.date >= start_date) & 
                    (df_filtered['Case Start Date'].dt.date <= end_date)
                ]
    
    # Enrich data with classifications and metrics
    df_enriched = enrich_data(df_filtered)
    
    # Store the enriched dataframe for use across tabs
    st.session_state.filtered_df = df_enriched
    
    #
    # DASHBOARD TAB
    #
    if selected == "Dashboard":
        st.title("📊 Executive Dashboard")
        
        # Top metrics row
        total_cases = len(df_enriched)
        sr_count = len(df_enriched[df_enriched['Type'] == 'SR'])
        incident_count = len(df_enriched[df_enriched['Type'] == 'Incident'])
        not_triaged = len(df_enriched[df_enriched['Status'] == 'Not Triaged'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            metric_card("Total Cases", total_cases)
        
        with col2:
            metric_card("Service Requests", sr_count)
        
        with col3:
            metric_card("Incidents", incident_count)
        
        with col4:
            metric_card("Not Triaged", not_triaged)
        
        # SLA Overview Chart
        st.subheader("📈 SLA Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create SLA data
            sla_counts = df_enriched['SLA Status'].value_counts().reset_index()
            sla_counts.columns = ['SLA Status', 'Count']
            
            # Create pie chart
            fig = px.pie(
                sla_counts, 
                values='Count', 
                names='SLA Status',
                color='SLA Status',
                color_discrete_map={
                    'Within SLA': '#2e7d32',
                    'At Risk': '#ff9800',
                    'Breached': '#c62828',
                    'Completed': '#1565c0',
                    'Unknown': '#9e9e9e'
                },
                hole=0.4
            )
            fig.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # SLA by user
            st.markdown("### SLA by User")
            
            user_sla = df_enriched.groupby('Current User Id')['SLA Status'].value_counts().unstack().fillna(0)
            
            if not user_sla.empty:
                # Calculate compliance percentage
                user_sla['Total'] = user_sla.sum(axis=1)
                if 'Within SLA' in user_sla.columns:
                    user_sla['Compliance %'] = round(user_sla['Within SLA'] / user_sla['Total'] * 100, 1)
                else:
                    user_sla['Compliance %'] = 0
                
                # Format for display
                user_sla_display = user_sla[['Compliance %']].sort_values('Compliance %', ascending=False)
                
                # Display as a table with progress bars
                for user, row in user_sla_display.iterrows():
                    compliance = row['Compliance %']
                    st.markdown(f"**{user}**")
                    st.progress(min(compliance / 100, 1.0))
                    st.markdown(f"{compliance}% compliant")
        
        # Recent Cases and Trend Charts
        st.subheader("📊 Recent Activity & Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Weekly Case Trend")
            
            # Prepare time series data
            if 'Case Start Date' in df_enriched.columns:
                # Create week column
                df_enriched['Week'] = df_enriched['Case Start Date'].dt.isocalendar().week
                
                # Group by week and type
                weekly_trend = df_enriched.groupby(['Week', 'Type']).size().unstack().fillna(0)
                
                # Create line chart
                if not weekly_trend.empty and 'SR' in weekly_trend.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=weekly_trend.index, 
                        y=weekly_trend['SR'], 
                        mode='lines+markers',
                        name='SR',
                        line=dict(color='#1976d2', width=3)
                    ))
                    
                    if 'Incident' in weekly_trend.columns:
                        fig.add_trace(go.Scatter(
                            x=weekly_trend.index, 
                            y=weekly_trend['Incident'], 
                            mode='lines+markers',
                            name='Incident',
                            line=dict(color='#ff9800', width=3)
                        ))
                    
                    fig.update_layout(
                        xaxis_title="Week Number",
                        yaxis_title="Number of Cases",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=20, r=20, t=30, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Insufficient data for trend analysis")
            else:
                st.info("Case Start Date not available for trend analysis")
        
        with col2:
            st.markdown("### Case Aging Distribution")
            
            # Create age bins
            age_bins = [0, 3, 7, 14, 30, float('inf')]
            age_labels = ['0-3 days', '4-7 days', '8-14 days', '15-30 days', '30+ days']
            
            df_enriched['Age Group'] = pd.cut(
                df_enriched['Age (Days)'], 
                bins=age_bins, 
                labels=age_labels, 
                right=False
            )
            
            age_dist = df_enriched['Age Group'].value_counts().sort_index()
            
            # Create bar chart
            fig = px.bar(
                x=age_dist.index, 
                y=age_dist.values,
                labels={'x': 'Age Group', 'y': 'Number of Cases'},
                color_discrete_sequence=['#1976d2']
            )
            fig.update_layout(
                xaxis_title="Age Group",
                yaxis_title="Number of Cases",
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Quick search and results
        st.subheader("🔎 Quick Search")
        
        search_col1, search_col2 = st.columns([1, 3])
        
        with search_col1:
            search_input = st.text_input("Enter SR or Incident Number:")
        
        with search_col2:
            if search_input.strip() and search_input.isdigit():
                search_number = int(search_input)
                search_results = df_enriched[df_enriched['Ticket Number'] == search_number]
                
                if not search_results.empty:
                    st.success(f"Found ticket #{search_number}")
                    shown_cols = ['Ticket Number', 'Case Id', 'Current User Id', 'Case Start Date', 'Status']
                    if 'SR Status' in search_results.columns:
                        shown_cols.append('SR Status')
                    if 'SLA Status' in search_results.columns:
                        shown_cols.append('SLA Status')
                    
                    st.dataframe(search_results[shown_cols])
                else:
                    st.warning(f"No results found for ticket #{search_number}")
    
    #
    # SR ANALYSIS TAB
    #
    elif selected == "SR Analysis":
        st.title("🔍 Detailed SR Analysis")
        
        # Filtering options
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_filter = st.selectbox(
                "Filter by Triage Status",
                ["All"] + df_enriched["Status"].dropna().unique().tolist()
            )
        
        with col2:
            type_filter = st.selectbox(
                "Filter by Type",
                ["All", "SR", "Incident"]
            )
        
        with col3:
            # SR Status filter (if available)
            if 'SR Status' in df_enriched.columns:
                sr_status_options = ["All"] + df_enriched['SR Status'].dropna().unique().tolist() + ["None"]
                sr_status_filter = st.selectbox("Filter by SR Status", sr_status_options)
            else:
                sr_status_filter = "All"
        
        with col4:
            # SLA Status filter
            sla_status_options = ["All"] + df_enriched['SLA Status'].dropna().unique().tolist()
            sla_filter = st.selectbox("Filter by SLA Status", sla_status_options)
        
        # Apply filters
        df_display = df_enriched.copy()
        
        if status_filter != "All":
            df_display = df_display[df_display["Status"] == status_filter]
        
        if type_filter != "All":
            df_display = df_display[df_display["Type"] == type_filter]
        
        if sr_status_filter != "All":
            if sr_status_filter == "None":
                df_display = df_display[df_display["SR Status"].isna()]
            else:
                df_display = df_display[df_display["SR Status"] == sr_status_filter]
        
        if sla_filter != "All":
            df_display = df_display[df_display["SLA Status"] == sla_filter]
        
        # Statistics and summary
        st.subheader("📊 Summary Analysis")
        
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.markdown("**🔸 Triage Status Count**")
            triage_summary = df_enriched['Status'].value_counts().rename_axis('Triage Status').reset_index(name='Count')
            triage_total = {'Triage Status': 'Total', 'Count': triage_summary['Count'].sum()}
            triage_df = pd.concat([triage_summary, pd.DataFrame([triage_total])], ignore_index=True)
            
            st.dataframe(
                triage_df.style.apply(
                    lambda x: ['background-color: #bbdefb; font-weight: bold' if x.name == len(triage_df)-1 else '' for _ in x],
                    axis=1
                )
            )
        
        with summary_col2:
            st.markdown("**🔹 SR vs Incident Count**")
            type_summary = df_enriched['Type'].value_counts().rename_axis('Type').reset_index(name='Count')
            type_total = {'Type': 'Total', 'Count': type_summary['Count'].sum()}
            type_df = pd.concat([type_summary, pd.DataFrame([type_total])], ignore_index=True)
            
            st.dataframe(
                type_df.style.apply(
                    lambda x: ['background-color: #bbdefb; font-weight: bold' if x.name == len(type_df)-1 else '' for _ in x],
                    axis=1
                )
            )
        
        with summary_col3:
            st.markdown("**🟢 SR Status Summary**")
            if 'SR Status' in df_enriched.columns:
                # Drop rows where SR Status is NaN
                df_status_valid = df_enriched.dropna(subset=['SR Status'])
                
                # All SR status count
                sr_all_counts = df_status_valid['SR Status'].value_counts().rename_axis('SR Status').reset_index(name='All SR Count')
                
                # Unique SRs
                sr_unique = df_status_valid.dropna(subset=['Ticket Number'])[['Ticket Number', 'SR Status']].drop_duplicates()
                sr_unique_counts = sr_unique['SR Status'].value_counts().rename_axis('SR Status').reset_index(name='Unique SR Count')
                
                # Merge both summaries
                merged_sr = pd.merge(sr_all_counts, sr_unique_counts, on='SR Status', how='outer').fillna(0)
                merged_sr[['All SR Count', 'Unique SR Count']] = merged_sr[['All SR Count', 'Unique SR Count']].astype(int)
                
                # Total row
                total_row = {
                    'SR Status': 'Total',
                    'All SR Count': merged_sr['All SR Count'].sum(),
                    'Unique SR Count': merged_sr['Unique SR Count'].sum()
                }
                
                sr_summary_df = pd.concat([merged_sr, pd.DataFrame([total_row])], ignore_index=True)
                
                # Display
                st.dataframe(
                    sr_summary_df.style.apply(
                        lambda x: ['background-color: #bbdefb; font-weight: bold' if x.name == len(sr_summary_df)-1 else '' for _ in x],
                        axis=1
                    )
                )
            else:
                st.info("Upload SR Status file to view this summary.")
        
        # SLA Analysis
        st.subheader("⏱️ SLA Analysis")
        
        sla_col1, sla_col2 = st.columns(2)
        
        with sla_col1:
            # SLA status counts
            sla_counts = df_display['SLA Status'].value_counts().reset_index()
            sla_counts.columns = ['SLA Status', 'Count']
            
            fig = px.bar(
                sla_counts,
                x='SLA Status',
                y='Count',
                color='SLA Status',
                color_discrete_map={
                    'Within SLA': '#2e7d32',
                    'At Risk': '#ff9800',
                    'Breached': '#c62828',
                    'Completed': '#1565c0',
                    'Unknown': '#9e9e9e'
                }
            )
            fig.update_layout(
                title="SLA Status Distribution",
                xaxis_title="SLA Status",
                yaxis_title="Number of Cases",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with sla_col2:
            # Average age by status
            if 'SR Status' in df_display.columns and 'Age (Days)' in df_display.columns:
                avg_age = df_display.groupby('SR Status')['Age (Days)'].mean().reset_index()
                avg_age.columns = ['SR Status', 'Average Age (Days)']
                avg_age['Average Age (Days)'] = avg_age['Average Age (Days)'].round(1)
                
                fig = px.bar(
                    avg_age,
                    x='SR Status',
                    y='Average Age (Days)',
                    color='Average Age (Days)',
                    color_continuous_scale=[(0, '#2e7d32'), (0.5, '#ff9800'), (1, '#c62828')]
                )
                fig.update_layout(
                    title="Average Age by SR Status",
                    xaxis_title="SR Status",
                    yaxis_title="Average Age (Days)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("SR Status data not available for this analysis.")
        
        # Detailed Results
        st.subheader("📋 Filtered Results")
        
        # Results count and download button
        results_col1, results_col2 = st.columns([3, 1])
        
        with results_col1:
            st.markdown(f"**Total Filtered Records:** {df_display.shape[0]}")
        
        with results_col2:
            if not df_display.empty:
                excel_data = generate_excel_download(df_display)
                st.download_button(
                    label="📥 Download Results",
                    data=excel_data,
                    file_name=f"sr_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        # Display data table with important columns
        important_cols = ['Ticket Number', 'Case Id', 'Current User Id', 'Case Start Date', 'Status', 'Type', 'Age (Days)', 'SLA Status']
        
        # Add SR Status columns if available
        if 'SR Status' in df_display.columns:
            important_cols.extend(['SR Status', 'Last Update'])
        
        # Ensure all columns exist
        display_cols = [col for col in important_cols if col in df_display.columns]
        
        # Custom formatting for SLA Status column
        def highlight_sla(val):
            if val == 'Within SLA':
                return 'background-color: #c8e6c9; color: #2e7d32'
            elif val == 'At Risk':
                return 'background-color: #ffecb3; color: #b17825'
            elif val == 'Breached':
                return 'background-color: #ffcdd2; color: #c62828'
            elif val == 'Completed':
                return 'background-color: #bbdefb; color: #1565c0'
            return ''
        
        # Apply styling
        styled_df = df_display[display_cols].style.applymap(
            highlight_sla, 
            subset=['SLA Status'] if 'SLA Status' in display_cols else []
        )
        
        st.dataframe(styled_df, height=400)
        
        # Note viewer
        if not df_display.empty:
            st.subheader("📝 Note Details")
            
            selected_case = st.selectbox(
                "Select a case to view notes:",
                df_display['Case Id'].tolist(),
                format_func=lambda x: f"Case #{x}"
            )
            
            if selected_case:
                case_data = df_display[df_display['Case Id'] == selected_case].iloc[0]
                
                note_col1, note_col2 = st.columns([1, 2])
                
                with note_col1:
                    st.markdown("**Case Details:**")
                    st.markdown(f"- **Case ID:** {case_data['Case Id']}")
                    st.markdown(f"- **Owner:** {case_data['Current User Id']}")
                    st.markdown(f"- **Start Date:** {case_data['Case Start Date'].strftime('%Y-%m-%d')}")
                    st.markdown(f"- **Age:** {case_data['Age (Days)']} days")
                    
                    if 'Ticket Number' in case_data and not pd.isna(case_data['Ticket Number']):
                        st.markdown(f"- **Ticket Number:** {int(case_data['Ticket Number'])}")
                    
                    if 'SR Status' in case_data and not pd.isna(case_data['SR Status']):
                        st.markdown(f"- **SR Status:** {case_data['SR Status']}")
                    
                    st.markdown(f"- **SLA Status:** {case_data['SLA Status']}")
                
                with note_col2:
                    st.markdown("**Last Note:**")
                    note_text = case_data['Last Note'] if not pd.isna(case_data['Last Note']) else "No notes available"
                    st.markdown(f"```\n{note_text}\n```")
                
                # Add quick actions
                st.markdown("**Quick Actions:**")
                action_col1, action_col2, action_col3 = st.columns(3)
                
                with action_col1:
                    st.button("Mark as Reviewed", key=f"review_{selected_case}")
                
                with action_col2:
                    st.button("Flag for Follow-up", key=f"flag_{selected_case}")
                
                with action_col3:
                    st.button("Add to Report", key=f"report_{selected_case}")
    
    #
    # TEAM PERFORMANCE TAB
    #
    elif selected == "Team Performance":
        st.title("👥 Team Performance Analysis")
        
        # Filter data for selected users
        if st.session_state.selected_users:
            team_df = df_enriched[df_enriched['Current User Id'].isin(st.session_state.selected_users)].copy()
        else:
            team_df = df_enriched.copy()
        
        # Time period selector
        period_options = ["Last 7 Days", "Last 30 Days", "All Time"]
        selected_period = st.selectbox("Select Time Period", period_options)
        
        # Filter by selected time period
        if 'Case Start Date' in team_df.columns:
            if selected_period == "Last 7 Days":
                cutoff_date = datetime.now() - timedelta(days=7)
                team_df = team_df[team_df['Case Start Date'] >= cutoff_date]
            elif selected_period == "Last 30 Days":
                cutoff_date = datetime.now() - timedelta(days=30)
                team_df = team_df[team_df['Case Start Date'] >= cutoff_date]
        
        # Team performance metrics
        st.subheader("📊 Team Performance Metrics")
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            total_cases = len(team_df)
            metric_card("Total Cases", total_cases)
        
        with metric_col2:
            avg_age = round(team_df['Age (Days)'].mean(), 1) if 'Age (Days)' in team_df.columns else 0
            metric_card("Average Age", f"{avg_age} days")
        
        with metric_col3:
            if 'SLA Status' in team_df.columns:
                sla_compliant = team_df[team_df['SLA Status'] == 'Within SLA'].shape[0]
                sla_percent = round((sla_compliant / total_cases * 100), 1) if total_cases > 0 else 0
                metric_card("SLA Compliance", f"{sla_percent}%")
            else:
                metric_card("SLA Compliance", "N/A")
        
        with metric_col4:
            if 'SR Status' in team_df.columns:
                completed = team_df[team_df['SR Status'].isin(['Completed', 'Closed'])].shape[0]
                completion_rate = round((completed / total_cases * 100), 1) if total_cases > 0 else 0
                metric_card("Completion Rate", f"{completion_rate}%")
            else:
                metric_card("Completion Rate", "N/A")
        
        # User comparison charts
        st.subheader("👤 User Performance Comparison")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Case volume by user
            user_counts = team_df.groupby('Current User Id').size().reset_index(name='Case Count')
            
            fig = px.bar(
                user_counts,
                x='Current User Id',
                y='Case Count',
                color='Case Count',
                color_continuous_scale=px.colors.sequential.Blues
            )
            fig.update_layout(
                title="Case Volume by User",
                xaxis_title="User",
                yaxis_title="Number of Cases"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with chart_col2:
            # SLA compliance by user
            if 'SLA Status' in team_df.columns:
                # Calculate SLA compliance percentage by user
                user_sla = team_df.groupby('Current User Id')['SLA Status'].apply(
                    lambda x: (x == 'Within SLA').mean() * 100
                ).reset_index(name='SLA Compliance %')
                
                fig = px.bar(
                    user_sla,
                    x='Current User Id',
                    y='SLA Compliance %',
                    color='SLA Compliance %',
                    color_continuous_scale=[(0, '#c62828'), (0.5, '#ff9800'), (1, '#2e7d32')]
                )
                fig.update_layout(
                    title="SLA Compliance by User",
                    xaxis_title="User",
                    yaxis_title="SLA Compliance %"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("SLA data not available for this analysis.")
        
        # Performance over time
        st.subheader("📈 Performance Over Time")
        
        if 'Case Start Date' in team_df.columns:
            # Create week and month columns
            team_df['Week'] = team_df['Case Start Date'].dt.isocalendar().week
            team_df['Month'] = team_df['Case Start Date'].dt.month
            team_df['Month Name'] = team_df['Case Start Date'].dt.month.apply(lambda x: calendar.month_abbr[x])
            
            # User selection for timeline
            selected_user = st.selectbox(
                "Select User for Timeline Analysis",
                ["All Users"] + list(team_df['Current User Id'].unique())
            )
            
            # Filter by selected user
            if selected_user != "All Users":
                timeline_df = team_df[team_df['Current User Id'] == selected_user]
            else:
                timeline_df = team_df
            
            # Time period selection for chart
            time_unit = st.radio("Time Unit", ["Weekly", "Monthly"], horizontal=True)
            
            if time_unit == "Weekly":
                # Weekly trend
                weekly_counts = timeline_df.groupby('Week').size().reset_index(name='Case Count')
                
                fig = px.line(
                    weekly_counts,
                    x='Week',
                    y='Case Count',
                    markers=True,
                    line_shape='linear'
                )
                fig.update_layout(
                    title=f"Weekly Case Volume - {selected_user}",
                    xaxis_title="Week Number",
                    yaxis_title="Number of Cases"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Monthly trend
                monthly_counts = timeline_df.groupby(['Month', 'Month Name']).size().reset_index(name='Case Count')
                monthly_counts = monthly_counts.sort_values('Month')
                
                fig = px.line(
                    monthly_counts,
                    x='Month Name',
                    y='Case Count',
                    markers=True,
                    line_shape='linear'
                )
                fig.update_layout(
                    title=f"Monthly Case Volume - {selected_user}",
                    xaxis_title="Month",
                    yaxis_title="Number of Cases",
                    xaxis={'categoryorder': 'array', 'categoryarray': [calendar.month_abbr[i] for i in range(1, 13)]}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Case Start Date not available for timeline analysis.")
        
        # Workload distribution
        st.subheader("⚖️ Workload Distribution Analysis")
        
        # Calculate workload metrics
        workload_df = team_df.groupby('Current User Id').agg({
            'Case Id': 'count',
            'Age (Days)': 'mean',
        }).reset_index()
        
        # Add SLA compliance if available
        if 'SLA Status' in team_df.columns:
            sla_by_user = team_df.groupby('Current User Id')['SLA Status'].apply(
                lambda x: (x == 'Within SLA').mean() * 100
            ).reset_index(name='SLA Compliance %')
            workload_df = workload_df.merge(sla_by_user, on='Current User Id')
        
        # Rename columns for display
        workload_df = workload_df.rename(columns={
            'Case Id': 'Case Count',
            'Age (Days)': 'Avg Age (Days)'
        })
        
        # Round numeric columns
        if 'Avg Age (Days)' in workload_df.columns:
            workload_df['Avg Age (Days)'] = workload_df['Avg Age (Days)'].round(1)
        if 'SLA Compliance %' in workload_df.columns:
            workload_df['SLA Compliance %'] = workload_df['SLA Compliance %'].round(1)
        
        # Display workload table
        st.dataframe(workload_df, height=300)
        
        # Add workload balance chart
        if len(workload_df) > 1:
            st.subheader("🔄 Workload Balance")
            
            # Calculate workload distribution statistics
            case_std = workload_df['Case Count'].std()
            case_mean = workload_df['Case Count'].mean()
            case_cv = case_std / case_mean if case_mean > 0 else 0
            
            # Create a gauge chart for workload balance
            balance_col1, balance_col2 = st.columns([1, 2])
            
            with balance_col1:
                st.markdown("### Workload Balance Score")
                
                # Calculate balance score (lower CV means better balance)
                balance_score = max(0, min(100, 100 * (1 - case_cv)))
                
                # Determine score color and message
                if balance_score >= 80:
                    score_color = "#2e7d32"  # Green
                    score_message = "Well Balanced"
                elif balance_score >= 60:
                    score_color = "#ff9800"  # Orange
                    score_message = "Moderately Balanced"
                else:
                    score_color = "#c62828"  # Red
                    score_message = "Needs Rebalancing"
                
                # Display score
                st.markdown(f"""
                <div style="text-align: center;">
                    <h1 style="color: {score_color}; font-size: 3em;">{balance_score:.1f}%</h1>
                    <p style="color: {score_color}; font-weight: bold;">{score_message}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with balance_col2:
                # Create recommendation based on workload balance
                st.markdown("### Workload Recommendations")
                
                if balance_score < 80:
                    # Find overloaded and underloaded users
                    workload_df['Difference from Average'] = workload_df['Case Count'] - case_mean
                    overloaded = workload_df[workload_df['Difference from Average'] > 5].sort_values('Case Count', ascending=False)
                    underloaded = workload_df[workload_df['Difference from Average'] < -5].sort_values('Case Count')
                    
                    if not overloaded.empty and not underloaded.empty:
                        st.markdown("#### Suggested Workload Transfers:")
                        
                        for _, over_user in overloaded.iterrows():
                            for _, under_user in underloaded.iterrows():
                                transfer_amount = min(
                                    over_user['Difference from Average'] / 2,
                                    abs(under_user['Difference from Average'])
                                )
                                transfer_amount = max(1, round(transfer_amount))
                                
                                st.markdown(f"""
                                * Transfer **{transfer_amount}** cases from **{over_user['Current User Id']}** ({over_user['Case Count']} cases) 
                                  to **{under_user['Current User Id']}** ({under_user['Case Count']} cases)
                                """)
                    else:
                        st.markdown("Workload is somewhat uneven, but no specific transfers are recommended.")
                else:
                    st.markdown("✅ **Workload is well balanced across the team.**")
                    st.markdown("Continue monitoring to maintain this balance.")
    
    #
    # SETTINGS TAB
    #
    elif selected == "Settings":
        st.title("⚙️ Configuration Settings")
        
        st.subheader("🛠️ System Configuration")
        
        # General configuration
        st.markdown("### General Settings")
        
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            default_theme = st.selectbox(
                "Default UI Theme",
                ["Light", "Dark", "Auto"]
            )
            
            data_refresh = st.selectbox(
                "Data Refresh Interval",
                ["Manual", "Hourly", "Daily", "Weekly"]
            )
        
        with config_col2:
            default_tab = st.selectbox(
                "Default Tab on Load",
                ["Dashboard", "SR Analysis", "Team Performance"]
            )
            
            language = st.selectbox(
                "Language",
                ["English", "Arabic", "French"]
            )
        
        # SLA configuration
        st.markdown("### SLA Configuration")
        
        sla_col1, sla_col2 = st.columns(2)
        
        with sla_col1:
            within_sla = st.number_input("Within SLA (days)", min_value=1, max_value=10, value=3)
            at_risk = st.number_input("At Risk (days)", min_value=within_sla, max_value=15, value=5)
        
        with sla_col2:
            breached = st.number_input("Breached (days)", min_value=at_risk, value=at_risk + 1)
            st.markdown("**Note:** Cases older than this will be marked as breached.")
        
        # User configuration
        st.subheader("👥 User Management")
        
        # Create a dataframe with sample users
        default_users = [
            {"Username": "ali.babiker", "Role": "SR Analyst", "Team": "Frontend Support"},
            {"Username": "anas.hasan", "Role": "SR Analyst", "Team": "Backend Support"},
            {"Username": "ahmed.mostafa", "Role": "Team Lead", "Team": "Frontend Support"}
        ]
        
        # Add more users from the data if available
        if st.session_state.main_df is not None:
            for user in st.session_state.main_df['Current User Id'].unique():
                if user not in [u["Username"] for u in default_users]:
                    default_users.append({
                        "Username": user,
                        "Role": "SR Analyst",
                        "Team": "General Support"
                    })
        
        user_df = pd.DataFrame(default_users)
        
        # Edit user information
        edited_user_df = st.data_editor(
            user_df,
            num_rows="dynamic",
            column_config={
                "Username": st.column_config.TextColumn("Username", help="User's login name"),
                "Role": st.column_config.SelectboxColumn(
                    "Role",
                    help="User's role in the system",
                    options=["SR Analyst", "Team Lead", "Manager", "Admin"]
                ),
                "Team": st.column_config.SelectboxColumn(
                    "Team",
                    help="User's team",
                    options=["Frontend Support", "Backend Support", "General Support", "Specialized Support"]
                )
            },
            use_container_width=True
        )
        
        st.button("Save User Configuration")
        
        # Export & Import settings
        st.subheader("💾 Export & Import Configuration")
        
        exp_col1, exp_col2 = st.columns(2)
        
        with exp_col1:
            st.download_button(
                label="Export Configuration",
                data="Sample configuration data",
                file_name=f"sr_analyzer_config_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        
        with exp_col2:
            st.file_uploader("Import Configuration", type=["json"])
        
        # Advanced settings
        st.subheader("🔧 Advanced Settings")
        
        adv_col1, adv_col2 = st.columns(2)
        
        with adv_col1:
            st.checkbox("Enable debug mode", value=False)
            st.checkbox("Cache data for faster performance", value=True)
        
        with adv_col2:
            st.checkbox("Send email notifications for breached SLAs", value=False)
            st.checkbox("Auto-generate weekly reports", value=False)
        
        # About section
        st.markdown("---")
        st.markdown("### About SR Analyzer Pro")
        st.markdown("""
        **Version:** 2.0.0
        
        **Developed by:** Your Organization
        
        **Last Updated:** April 2025
        
        This application helps you manage and analyze Service Requests and Incidents effectively.
        For support or feedback, please contact your system administrator.
        """)
                
