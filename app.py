```python
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
# New session state for tracked rows
if 'tracked_rows' not in st.session_state:
    st.session_state.tracked_rows = []

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
    match = re.search(r'(tkt|sr|inc|ticket|مرجعي|incident|اس ار|انسدنت)[\s\S]{0,50}?(\d{4,})', note_lower)
        
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

# Function to handle row selection for tracking
def track_row(row_data):
    case_id = row_data['Case Id']
    
    # Check if row is already tracked
    if case_id in [row['Case Id'] for row in st.session_state.tracked_rows]:
        # Remove from tracked rows
        st.session_state.tracked_rows = [row for row in st.session_state.tracked_rows if row['Case Id'] != case_id]
    else:
        # Add to tracked rows
        st.session_state.tracked_rows.append(row_data.to_dict())
    
    # Force rerun to update UI
    st.rerun()

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
    
    # Display last upload time
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
    - Team performance metrics
    - Track unresolved SRs in a dedicated tab
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
        options=["Dashboard", "SR Analysis", "Not Resolved SR", "Team Performance", "Settings"],
        icons=["speedometer2", "kanban", "clipboard-check", "people", "gear"],
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
        
        # Display last update time
        st.markdown(f"**Last data update:** {st.session_state.last_upload_time}")
        
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
        
        # SR Status Overview Chart
        st.subheader("📈 SR Status Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create status data if SR status data is available
            if 'SR Status' in df_enriched.columns:
                status_counts = df_enriched['SR Status'].value_counts().reset_index()
                status_counts.columns = ['SR Status', 'Count']
                
                # Create pie chart
                fig = px.pie(
                    status_counts, 
                    values='Count', 
                    names='SR Status',
                    hole=0.4
                )
                fig.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
                    margin=dict(l=20, r=20, t=30, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("SR Status data not available. Please upload SR Status file.")
        
        with col2:
            # Status by user
            st.markdown("### Status by User")
            
            if 'SR Status' in df_enriched.columns:
                user_status = df_enriched.groupby('Current User Id')['SR Status'].value_counts().unstack().fillna(0)
                
                if not user_status.empty:
                    # Calculate total cases per user
                    user_status['Total'] = user_status.sum(axis=1)
                    
                    # Format for display
                    user_status_display = user_status.sort_values('Total', ascending=False)
                    
                    # Display as a table
                    st.dataframe(user_status_display)
            else:
                st.info("SR Status data not available")
        
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
                    
                    st.dataframe(search_results[shown_cols])
                else:
                    st.warning(f"No results found for ticket #{search_number}")
    
    #
    # SR ANALYSIS TAB
    #
    elif selected == "SR Analysis":
        st.title("🔍 Detailed SR Analysis")
        
        # Display last update time
        st.markdown(f"**Last data update:** {st.session_state.last_upload_time}")
        
        # Filtering options
        col1, col2, col3 = st.columns(3)
        
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
        
        # Display data table with important columns and checkboxes
        important_cols = ['Last Note', 'Ticket Number', 'Case Id', 'Current User Id', 'Case Start Date', 'Status', 'Type', 'Age (Days)']
        
        # Add SR Status columns if available
        if 'SR Status' in df_display.columns:
            important_cols.extend(['SR Status', 'Last Update'])
        
        # Ensure all columns exist
        display_cols = [col for col in important_cols if col in df_display.columns]
        
        # Create checkboxes for each row
        for i, row in df_display.iterrows():
            col1, col2 = st.columns([0.5, 11.5])
            
            with col1:
                # Check if this row is already tracked
                is_tracked = row['Case Id'] in [tracked_row['Case Id'] for tracked_row in st.session_state.tracked_rows]
                
                # Create checkbox for tracking
                if st.checkbox("", value=is_tracked, key=f"track_sr_{i}_{row['Case Id']}"):
                    if not is_tracked:
                        # Add to tracked rows if not already tracked
                        track_row(row)
                else:
                    if is_tracked:
                        # Remove from tracked rows if it was tracked
                        track_row(row)
            
            with col2:
                # Display row data
                st.write(f"**Case ID:** {row['Case Id']}")
                st.write(f"**Ticket Number:** {int(row['Ticket Number']) if not pd.isna(row['Ticket Number']) else 'N/A'}")
                st.write(f"**Owner:** {row['Current User Id']}")
                st.write(f"**Status:** {row['Status']}")
                if 'SR Status' in row and not pd.isna(row['SR Status']):
                    st.write(f"**SR Status:** {row['SR Status']}")
                st.write(f"**Age:** {row['Age (Days)']} days")
                st.markdown("---")
        
        # Note viewer
        if not df_display.empty:
            st.subheader("📝 Note Details")
            
            selected_case = st.selectbox(
                "Select a case to view notes:",
                df_display['Case Id'].tolist()
            )
            
            if selected_case:
                case_row = df_display[df_display['Case Id'] == selected_case].iloc[0]
                
                # Display case details
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.markdown("### Case Details")
                    st.write(f"**Case ID:** {case_row['Case Id']}")
                    st.write(f"**Owner:** {case_row['Current User Id']}")
                    st.write(f"**Start Date:** {case_row['Case Start Date'].strftime('%Y-%m-%d')}")
                    st.write(f"**Age:** {case_row['Age (Days)']} days")
                
                with detail_col2:
                    st.markdown("### Ticket Details")
                    if not pd.isna(case_row['Ticket Number']):
                        st.write(f"**Ticket Number:** {int(case_row['Ticket Number'])}")
                        st.write(f"**Type:** {case_row['Type']}")
                        
                        if 'SR Status' in case_row and not pd.isna(case_row['SR Status']):
                            st.write(f"**SR Status:** {case_row['SR Status']}")
                            
                            if 'Last Update' in case_row and not pd.isna(case_row['Last Update']):
                                st.write(f"**Last Update:** {case_row['Last Update']}")
                    else:
                        st.write("No ticket information available")
                
                # Display the full note
                st.markdown("### Last Note")
                if 'Last Note' in case_row and not pd.isna(case_row['Last Note']):
                    st.markdown(f"```\n{case_row['Last Note']}\n```")
                else:
                    st.info("No notes available for this case")
                
                # Add action buttons
                action_col1, action_col2, action_col3 = st.columns(3)
                
                with action_col1:
                    if st.button("📝 Add to Tracked", key=f"add_tracked_{selected_case}"):
                        track_row(case_row)
                        st.success(f"Case {selected_case} added to tracked items")
                
                with action_col2:
                    if 'SR Status' in df_enriched.columns:
                        if st.button("🔄 Refresh SR Status", key=f"refresh_sr_{selected_case}"):
                            st.info("SR Status refresh functionality would go here")
                
                with action_col3:
                    if st.button("📤 Export Case Details", key=f"export_{selected_case}"):
                        case_details = df_display[df_display['Case Id'] == selected_case]
                        excel_data = generate_excel_download(case_details)
                        st.download_button(
                            label="Download Case Details",
                            data=excel_data,
                            file_name=f"case_{selected_case}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"dl_case_{selected_case}"
                        )
    
    #
    # NOT RESOLVED SR TAB
    #
    elif selected == "Not Resolved SR":
        st.title("📋 Tracked Service Requests")
        
        # Display tracked SRs
        if not st.session_state.tracked_rows:
            st.info("No service requests are currently being tracked. Add SRs from the SR Analysis tab.")
        else:
            # Convert tracked rows to dataframe
            tracked_df = pd.DataFrame(st.session_state.tracked_rows)
            
            # Display statistics
            st.subheader("📊 Tracked SR Statistics")
            
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            
            with stat_col1:
                metric_card("Total Tracked Items", len(tracked_df))
            
            with stat_col2:
                sr_count = len(tracked_df[tracked_df['Type'] == 'SR'])
                metric_card("Service Requests", sr_count)
            
            with stat_col3:
                incident_count = len(tracked_df[tracked_df['Type'] == 'Incident'])
                metric_card("Incidents", incident_count)
            
            # Download button
            st.download_button(
                label="📥 Download Tracked Items",
                data=generate_excel_download(tracked_df),
                file_name=f"tracked_items_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            # Display tracked items
            st.subheader("📋 Tracked Items List")
            
            for i, row in tracked_df.iterrows():
                col1, col2 = st.columns([0.5, 11.5])
                
                with col1:
                    if st.button("❌", key=f"remove_{i}_{row['Case Id']}"):
                        # Remove from tracked rows
                        track_row(row)
                
                with col2:
                    # Display row data with status badge
                    status_badge = ""
                    if 'SR Status' in row and not pd.isna(row['SR Status']):
                        status_class = "badge-pending"
                        
                        if row['SR Status'].lower() in ['completed', 'resolved', 'closed']:
                            status_class = "badge-complete"
                        elif row['SR Status'].lower() in ['in progress', 'assigned', 'working']:
                            status_class = "badge-in-progress"
                        elif row['SR Status'].lower() in ['cancelled', 'rejected']:
                            status_class = "badge-cancelled"
                        
                        status_badge = f'<span class="status-badge {status_class}">{row["SR Status"]}</span>'
                    
                    st.markdown(f"**Case ID:** {row['Case Id']} {status_badge}", unsafe_allow_html=True)
                    st.write(f"**Ticket Number:** {int(row['Ticket Number']) if not pd.isna(row['Ticket Number']) else 'N/A'}")
                    st.write(f"**Owner:** {row['Current User Id']}")
                    st.write(f"**Age:** {row['Age (Days)']} days")
                    
                    # Show note preview
                    if 'Last Note' in row and not pd.isna(row['Last Note']):
                        note_preview = row['Last Note'][:100] + "..." if len(row['Last Note']) > 100 else row['Last Note']
                        st.write(f"**Note:** {note_preview}")
                    
                    st.markdown("---")
    
    #
    # TEAM PERFORMANCE TAB
    #
    elif selected == "Team Performance":
        st.title("👥 Team Performance Metrics")
        
        # Display team summary
        st.subheader("📊 Team Summary")
        
        # Calculate metrics by user
        if not df_enriched.empty:
            # Group by user
            user_stats = df_enriched.groupby('Current User Id').agg(
                Total_Cases=('Case Id', 'count'),
                SR_Count=('Type', lambda x: (x == 'SR').sum()),
                Incident_Count=('Type', lambda x: (x == 'Incident').sum()),
                Not_Triaged=('Status', lambda x: (x == 'Not Triaged').sum()),
                Average_Age=('Age (Days)', 'mean')
            ).reset_index()
            
            # Sort by total cases
            user_stats = user_stats.sort_values('Total_Cases', ascending=False)
            
            # Calculate percentages
            user_stats['SR_Percent'] = (user_stats['SR_Count'] / user_stats['Total_Cases'] * 100).round(1)
            user_stats['Incident_Percent'] = (user_stats['Incident_Count'] / user_stats['Total_Cases'] * 100).round(1)
            user_stats['Not_Triaged_Percent'] = (user_stats['Not_Triaged'] / user_stats['Total_Cases'] * 100).round(1)
            
            # Format average age
            user_stats['Average_Age'] = user_stats['Average_Age'].round(1)
            
            # Display team stats
            st.dataframe(user_stats.style.background_gradient(subset=['Total_Cases', 'SR_Count', 'Not_Triaged_Percent'], cmap='YlOrRd'))
            
            # Create charts
            st.subheader("📈 Performance Visualization")
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.markdown("### Case Distribution by User")
                
                # Create case distribution chart
                fig = px.bar(
                    user_stats,
                    x='Current User Id',
                    y=['SR_Count', 'Incident_Count', 'Not_Triaged'],
                    title='Case Distribution by User',
                    labels={'value': 'Number of Cases', 'Current User Id': 'User', 'variable': 'Case Type'},
                    color_discrete_map={
                        'SR_Count': '#1976d2',
                        'Incident_Count': '#ff9800',
                        'Not_Triaged': '#e53935'
                    }
                )
                
                fig.update_layout(
                    xaxis_title="User",
                    yaxis_title="Number of Cases",
                    legend_title="Case Type",
                    barmode='stack'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with chart_col2:
                st.markdown("### Average Case Age by User")
                
                # Create average age chart
                fig = px.bar(
                    user_stats,
                    x='Current User Id',
                    y='Average_Age',
                    title='Average Case Age by User',
                    labels={'Average_Age': 'Average Age (Days)', 'Current User Id': 'User'},
                    color='Average_Age',
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(
                    xaxis_title="User",
                    yaxis_title="Average Age (Days)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # User specific metrics
            st.subheader("🔍 User Specific Metrics")
            
            selected_user = st.selectbox(
                "Select a user for detailed metrics:",
                user_stats['Current User Id'].tolist()
            )
            
            if selected_user:
                # Filter data for selected user
                user_data = df_enriched[df_enriched['Current User Id'] == selected_user]
                
                # Display user metrics
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    metric_card("Total Cases", len(user_data))
                
                with metric_col2:
                    sr_count = len(user_data[user_data['Type'] == 'SR'])
                    metric_card("Service Requests", sr_count)
                
                with metric_col3:
                    incident_count = len(user_data[user_data['Type'] == 'Incident'])
                    metric_card("Incidents", incident_count)
                
                with metric_col4:
                    not_triaged = len(user_data[user_data['Status'] == 'Not Triaged'])
                    metric_card("Not Triaged", not_triaged)
                
                # Create detailed charts for the user
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.markdown("### Status Distribution")
                    
                    if 'SR Status' in user_data.columns:
                        status_counts = user_data['SR Status'].value_counts().reset_index()
                        status_counts.columns = ['SR Status', 'Count']
                        
                        fig = px.pie(
                            status_counts,
                            values='Count',
                            names='SR Status',
                            hole=0.4,
                            title=f'SR Status Distribution for {selected_user}'
                        )
                        
                        fig.update_layout(
                            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("SR Status data not available")
                
                with detail_col2:
                    st.markdown("### Case Age Distribution")
                    
                    age_counts = user_data['Age Group'].value_counts().sort_index().reset_index()
                    age_counts.columns = ['Age Group', 'Count']
                    
                    fig = px.bar(
                        age_counts,
                        x='Age Group',
                        y='Count',
                        title=f'Case Age Distribution for {selected_user}',
                        color='Count',
                        color_continuous_scale='Blues'
                    )
                    
                    fig.update_layout(
                        xaxis_title="Age Group",
                        yaxis_title="Number of Cases"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display user's recent cases
                st.subheader(f"Recent Cases for {selected_user}")
                
                # Sort by case start date
                recent_cases = user_data.sort_values('Case Start Date', ascending=False).head(10)
                
                # Display as a table
                if not recent_cases.empty:
                    display_cols = ['Case Id', 'Case Start Date', 'Status', 'Type', 'Ticket Number', 'Age (Days)']
                    
                    if 'SR Status' in recent_cases.columns:
                        display_cols.append('SR Status')
                    
                    st.dataframe(recent_cases[display_cols])
                else:
                    st.info(f"No cases found for {selected_user}")
                
                # Export user data button
                st.download_button(
                    label=f"📥 Download {selected_user}'s Data",
                    data=generate_excel_download(user_data),
                    file_name=f"{selected_user}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    #
    # SETTINGS TAB
    #
    elif selected == "Settings":
        st.title("⚙️ Application Settings")
        
        st.subheader("🧾 Data Management")
        
        # Clear data option
        if st.button("🗑️ Clear All Data"):
            # Add confirmation
            if st.checkbox("Are you sure? This will remove all loaded data."):
                st.session_state.data_loaded = False
                st.session_state.main_df = None
                st.session_state.sr_df = None
                st.session_state.filtered_df = None
                st.session_state.last_upload_time = None
                st.session_state.tracked_rows = []
                st.success("All data has been cleared!")
                st.rerun()
        
        # Clear only tracked rows
        if st.button("🧹 Clear Tracked Items"):
            if st.checkbox("Are you sure? This will remove all tracked items."):
                st.session_state.tracked_rows = []
                st.success("All tracked items have been cleared!")
        
        st.subheader("🔧 Application Information")
        
        # App information
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.markdown("### About This Application")
            st.markdown("""
            **SR Analyzer Pro** is a specialized tool designed to help teams manage and analyze Service Requests and Incidents efficiently.
            
            **Version:** 1.0.0
            **Last Updated:** April 2025
            """)
        
        with info_col2:
            st.markdown("### Usage Statistics")
            if st.session_state.main_df is not None:
                st.markdown(f"**Total Records:** {len(st.session_state.main_df)}")
                st.markdown(f"**Tracked Items:** {len(st.session_state.tracked_rows)}")
                
                if st.session_state.last_upload_time:
                    st.markdown(f"**Data Last Uploaded:** {st.session_state.last_upload_time}")
            else:
                st.info("No data has been loaded yet.")
        
        # Export configuration
        st.subheader("📤 Export Configuration")
        
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            st.markdown("### Export Options")
            
            export_format = st.selectbox(
                "Export Format",
                ["Excel (.xlsx)", "CSV (.csv)"]
            )
            
            include_charts = st.checkbox("Include charts in export", value=True)
        
        with export_col2:
            st.markdown("### Export Content")
            
            export_content = st.multiselect(
                "Select content to export",
                ["Main Data", "SR Status Data", "Tracked Items", "Analysis Results"],
                default=["Main Data"]
            )
            
            if st.button("Generate Export"):
                st.info("Export functionality would be implemented here")

# Run the application
if __name__ == "__main__":
    pass