import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import base64
from datetime import datetime, timedelta
import pytz
from streamlit_option_menu import option_menu

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
    .badge-breach {
        background-color: #ff5252;
        color: white;
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
    h1, h2, h3 {
        color: #1565c0;
    }
    /* Custom styling for multi-select dataframe */
    div[data-testid="stDataFrame"] table {
        width: 100%;
    }
    div[data-testid="stDataFrame"] th:first-child, 
    div[data-testid="stDataFrame"] td:first-child {
        width: 50px !important;
        min-width: 50px !important;
        max-width: 50px !important;
    }
    .st-ch {
        font-size: 14px;
    }
    .action-button {
        background-color: #1976d2;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 5px 10px;
        cursor: pointer;
        font-size: 14px;
        margin-bottom: 10px;
    }
    .action-button:hover {
        background-color: #1565c0;
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
if 'selected_case_ids' not in st.session_state:
    st.session_state.selected_case_ids = []

# Function to load and process data
@st.cache_data
def load_data(file):
    return pd.read_excel(file)

# Function to process main dataframe
def process_main_df(df):
    # Ensure date columns are in datetime format
    date_columns = ['Case Start Date', 'Last Note Date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%d/%m/%Y", errors='coerce')
    return df
    
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
        ticket_type = "SR" if 14000 <= ticket_num <= 17000 else "Incident"
        return "Pending SR/Incident", ticket_num, ticket_type
    
    return "Not Triaged", None, None

# Function to calculate case age in days
def calculate_age(start_date):
    if pd.isna(start_date):
        return None
    return (datetime.now() - start_date).days

# Function to determine if a note was created today
def is_created_today(date_value):
    if pd.isna(date_value):
        return False
    today = datetime.now().date()
    note_date = date_value.date() if isinstance(date_value, datetime) else date_value
    return note_date == today

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

# Sidebar - File Upload Section
with st.sidebar:
    st.title("📊 SR Analyzer Pro TeeesT")
    st.markdown("---")

    st.subheader("📁 Data Import")
    uploaded_file = st.file_uploader("Upload Main Excel File (.xlsx)", type=["xlsx","xls"])
    sr_status_file = st.file_uploader("Upload SR Status Excel (optional)", type=["xlsx","xls"])
    
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
    abu_dhabi_tz = pytz.timezone('Asia/Dubai')
    st.session_state.last_upload_time = datetime.now(abu_dhabi_tz).strftime("%Y-%m-%d %H:%M:%S")
    if st.session_state.last_upload_time:
        st.info(f"Last upload: {st.session_state.last_upload_time}")
    
    st.markdown("---")
    
    # Filters section
    if st.session_state.data_loaded:
        st.subheader("🔍 Filters")
        
        # Get all users
        df_main = st.session_state.main_df.copy()
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
        
        # Date range filter
        if 'Case Start Date' in df_main.columns:
            min_date = df_main['Case Start Date'].min().date()
            max_date = df_main['Case Start Date'].max().date()
            
            date_range = st.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )

# Main content
if not st.session_state.data_loaded:
    st.title("📊 SR Analyzer Pro")
    st.markdown("""
    ### Welcome to the SR Analyzer Pro!
    
    This application helps you analyze Service Requests and Incidents efficiently.
    
    To get started:
    1. Upload your main Excel file using the sidebar
    2. Optionally upload SR status file for enhanced analysis
    3. Use the application to analyze and export your data
    
    **Features:**
    - Advanced filtering and search
    - Detailed SR Analysis
    - SLA Breach monitoring
    - Today's new incidents and SRs
    """)
else:
    # Process and filter data
    df_main = st.session_state.main_df.copy()
    
    # Apply user filters
    if st.session_state.selected_users:
        df_filtered = df_main[df_main['Current User Id'].isin(st.session_state.selected_users)].copy()
    else:
        df_filtered = df_main.copy()
    
    # Apply date filter if date range is selected
    if 'Case Start Date' in df_filtered.columns and 'date_range' in locals():
        start_date, end_date = date_range
        df_filtered = df_filtered[
            (df_filtered['Case Start Date'].dt.date >= start_date) & 
            (df_filtered['Case Start Date'].dt.date <= end_date)
        ]
    
    # Prepare tab interface
    selected = option_menu(
        menu_title=None,
        options=["SR Analysis", "SLA Breach", "Today's SR/Incidents"],
        icons=["kanban", "exclamation-triangle", "calendar-date"],
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
        
        # Determine if note was created today
        if 'Last Note Date' in df_enriched.columns:
            df_enriched['Created Today'] = df_enriched['Last Note Date'].apply(is_created_today)
        
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
                sr_df[['Service Request', 'SR Status', 'Last Update', 'Breach Passed']],
                how='left',
                left_on='Ticket Number',
                right_on='Service Request'
            ).drop(columns=['Service Request'])

            # After merging with SR status data
        if 'Breach Date' in df_enriched.columns:
            df_enriched['Breach Date'] = pd.to_datetime(df_enriched['Breach Date'], errors='coerce')
        return df_enriched
    
    # Enrich data with classifications and metrics
    df_enriched = enrich_data(df_filtered)
    
    # Store the enriched dataframe for use across tabs
    st.session_state.filtered_df = df_enriched
    
    #
    # SR ANALYSIS TAB
    #
    if selected == "SR Analysis":
        st.title("🔍 SR Analysis")
        
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
        
        # Display data table with important columns
        important_cols = ['Last Note', 'Case Id', 'Current User Id', 'Case Start Date', 'Status', 'Type', 'Ticket Number']
        
        # Add SR Status columns if available
        if 'SR Status' in df_display.columns:
            important_cols.extend(['SR Status', 'Last Update'])
            if 'Breach Passed' in df_display.columns:
                important_cols.append('Breach Passed')
        
        # Ensure all columns exist
        display_cols = [col for col in important_cols if col in df_display.columns]
        
        # Prepare dataframe for display
        if not df_display.empty:
            # Display with st.dataframe
            st.dataframe(df_display[display_cols], hide_index=True)
        
        # Note viewer
        st.subheader("📝 Note Details")
        
        selected_case = st.selectbox(
            "Select a case to view notes:",
            df_display['Case Id'].tolist()
        )
        
        if selected_case:
            case_row = df_display[df_display['Case Id'] == selected_case].iloc[0]
            
            # Display case details in a table
            case_details = {
                "Field": ["Case ID", "Owner", "Start Date", "Age", "Ticket Number", "Type"],
                "Value": [
                    case_row['Case Id'],
                    case_row['Current User Id'],
                    case_row['Case Start Date'].strftime('%Y-%m-%d'),
                    f"{case_row['Age (Days)']} days",
                    int(case_row['Ticket Number']) if not pd.isna(case_row['Ticket Number']) else 'N/A',
                    case_row['Type'] if not pd.isna(case_row['Type']) else 'N/A'
                ]
            }
            
            # Add SR Status if available
            if 'SR Status' in case_row and not pd.isna(case_row['SR Status']):
                case_details["Field"].append("SR Status")
                case_details["Value"].append(case_row['SR Status'])
                
                if 'Last Update' in case_row and not pd.isna(case_row['Last Update']):
                    case_details["Field"].append("Last Update")
                    case_details["Value"].append(case_row['Last Update'])
                
                if 'Breach Passed' in case_row:
                    case_details["Field"].append("SLA Breach")
                    case_details["Value"].append("Yes ⚠️" if case_row['Breach Passed'] == True else "No")
            
            # Display as a table
            st.table(pd.DataFrame(case_details))
            
            # Display the full note
            st.markdown("### Last Note")
            if 'Last Note' in case_row and not pd.isna(case_row['Last Note']):
                st.text_area("Note Content", case_row['Last Note'], height=200)
            else:
                st.info("No notes available for this case")
            
            # Download button for case details
            excel_data = generate_excel_download(df_display[df_display['Case Id'] == selected_case])
            st.download_button(
                label="📥 Download Case Details",
                data=excel_data,
                file_name=f"case_{selected_case}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
    #
    # SLA BREACH TAB
    #
    elif selected == "SLA Breach":
        st.title("⚠️ SLA Breach Analysis")
        
        # Check if SR data is available
        if st.session_state.sr_df is None:
            st.warning("Please upload SR Status Excel file to view SLA breach information.")
        else:
            # Filter to get only breach cases
            if 'Breach Passed' in df_enriched.columns:
                breach_df = df_enriched[df_enriched['Breach Passed'] == True].copy()
                
                # Display summary statistics
                st.subheader("📊 SLA Breach Summary")
                
                # Statistics cards
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    breach_count = len(breach_df)
                    st.markdown(f'<p class="metric-value">{breach_count}</p>', unsafe_allow_html=True)
                    st.markdown('<p class="metric-label">Total SLA Breaches</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    if 'SR Status' in breach_df.columns:
                        open_breaches = len(breach_df[breach_df['SR Status'].isin(['Open', 'In Progress', 'Pending'])])
                        st.markdown(f'<p class="metric-value">{open_breaches}</p>', unsafe_allow_html=True)
                        st.markdown('<p class="metric-label">Open Breached Cases</p>', unsafe_allow_html=True)
                    else:
                        st.markdown('<p class="metric-value">N/A</p>', unsafe_allow_html=True)
                        st.markdown('<p class="metric-label">Open Breached Cases</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    if 'Current User Id' in breach_df.columns:
                        user_breach_count = len(breach_df['Current User Id'].unique())
                        st.markdown(f'<p class="metric-value">{user_breach_count}</p>', unsafe_allow_html=True)
                        st.markdown('<p class="metric-label">Users Affected</p>', unsafe_allow_html=True)
                    else:
                        st.markdown('<p class="metric-value">N/A</p>', unsafe_allow_html=True)
                        st.markdown('<p class="metric-label">Users Affected</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Filtering options for breach data
                st.subheader("🔍 Filter SLA Breaches")
                
                filter_col1, filter_col2 = st.columns(2)
                
                with filter_col1:
                    if 'SR Status' in breach_df.columns:
                        breach_status_options = ["All"] + breach_df['SR Status'].dropna().unique().tolist()
                        breach_status_filter = st.selectbox("Filter by Status", breach_status_options)
                    else:
                        breach_status_filter = "All"
                
                with filter_col2:
                    if 'Current User Id' in breach_df.columns:
                        breach_user_options = ["All"] + breach_df['Current User Id'].dropna().unique().tolist()
                        breach_user_filter = st.selectbox("Filter by User", breach_user_options)
                    else:
                        breach_user_filter = "All"
                
                # Apply filters
                filtered_breach_df = breach_df.copy()
                
                if breach_status_filter != "All" and 'SR Status' in filtered_breach_df.columns:
                    filtered_breach_df = filtered_breach_df[filtered_breach_df['SR Status'] == breach_status_filter]
                
                if breach_user_filter != "All" and 'Current User Id' in filtered_breach_df.columns:
                    filtered_breach_df = filtered_breach_df[filtered_breach_df['Current User Id'] == breach_user_filter]
                
                # Breach data table
                st.subheader("📋 SLA Breach Details")
                
                if filtered_breach_df.empty:
                    st.info("No SLA breaches found matching the current filters.")
                else:
                    # Results count and download button
                    results_col1, results_col2 = st.columns([3, 1])
                    
                    with results_col1:
                        st.markdown(f"**Total Breached Records:** {filtered_breach_df.shape[0]}")
                    
                    with results_col2:
                        excel_data = generate_excel_download(filtered_breach_df)
                        st.download_button(
                            label="📥 Download Breaches",
                            data=excel_data,
                            file_name=f"sla_breaches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    # Important columns for breach display
                    breach_display_cols = ['Case Id','Last Note', 'Current User Id', 'Case Start Date', 'Type', 'Ticket Number', 'SR Status', 'Breach Date']
                    
                    # Ensure all columns exist
                    breach_display_cols = [col for col in breach_display_cols if col in filtered_breach_df.columns]
                    
                    # Display breach data
                    st.dataframe(filtered_breach_df[breach_display_cols], hide_index=True)
                    
                    # Detailed breach case viewer
                    st.subheader("🔍 Breach Case Details")
                    
                    selected_breach_case = st.selectbox(
                        "Select a breached case to view details:",
                        filtered_breach_df['Case Id'].tolist()
                    )
                    
                    if selected_breach_case:
                        breach_row = filtered_breach_df[filtered_breach_df['Case Id'] == selected_breach_case].iloc[0]
                        
                        # Display case details in a table
                        breach_details = {
                            "Field": ["Case ID", "Owner", "Start Date", "Age", "Ticket Number", "Type"],
                            "Value": [
                                breach_row['Case Id'],
                                breach_row['Current User Id'],
                                breach_row['Case Start Date'].strftime('%Y-%m-%d'),
                                f"{breach_row['Age (Days)']} days",
                                int(breach_row['Ticket Number']) if not pd.isna(breach_row['Ticket Number']) else 'N/A',
                                breach_row['Type'] if not pd.isna(breach_row['Type']) else 'N/A'
                            ]
                        }
                        
                        # Add SR Status if available
                        if 'SR Status' in breach_row and not pd.isna(breach_row['SR Status']):
                            breach_details["Field"].append("SR Status")
                            breach_details["Value"].append(breach_row['SR Status'])
                            
                            if 'Last Update' in breach_row and not pd.isna(breach_row['Last Update']):
                                breach_details["Field"].append("Last Update")
                                breach_details["Value"].append(breach_row['Last Update'])
                        
                        # Display as a table
                        st.table(pd.DataFrame(breach_details))
                        
                        # Display the full note
                        st.markdown("### Last Note")
                        if 'Last Note' in breach_row and not pd.isna(breach_row['Last Note']):
                            st.text_area("Note Content", breach_row['Last Note'], height=200)
                        else:
                            st.info("No notes available for this case")
            else:
                st.warning("SLA Breach information not available. Please ensure your SR Status file contains breach data.")
    
    #
    # TODAY'S SR/INCIDENTS TAB
    #
    elif selected == "Today's SR/Incidents":
        st.title("📅 Today's SR/Incidents")
        
        # Filter to get today's entries
        if 'Created Today' in df_enriched.columns:
            today_df = df_enriched[df_enriched['Created Today'] == True].copy()
            
            # Display summary statistics
            st.subheader("📊 Today's Activity Summary")
            
            # Statistics cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                today_count = len(today_df)
                st.markdown(f'<p class="metric-value">{today_count}</p>', unsafe_allow_html=True)
                st.markdown('<p class="metric-label">Total Today\'s Activities</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                sr_count = len(today_df[today_df['Type'] == 'SR'])
                st.markdown(f'<p class="metric-value">{sr_count}</p>', unsafe_allow_html=True)
                st.markdown('<p class="metric-label">Today\'s SRs</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                incident_count = len(today_df[today_df['Type'] == 'Incident'])
                st.markdown(f'<p class="metric-value">{incident_count}</p>', unsafe_allow_html=True)
                st.markdown('<p class="metric-label">Today\'s Incidents</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Filtering options for today's data
            st.subheader("🔍 Filter Today's Activities")
            
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                today_type_options = ["All", "SR", "Incident"]
                today_type_filter = st.selectbox("Filter by Type", today_type_options)
            
            with filter_col2:
                if 'Current User Id' in today_df.columns:
                    today_user_options = ["All"] + today_df['Current User Id'].dropna().unique().tolist()
                    today_user_filter = st.selectbox("Filter by User", today_user_options)
                else:
                    today_user_filter = "All"
            
            # Apply filters
            filtered_today_df = today_df.copy()
            
            if today_type_filter != "All":
                filtered_today_df = filtered_today_df[filtered_today_df['Type'] == today_type_filter]
            
            if today_user_filter != "All" and 'Current User Id' in filtered_today_df.columns:
                filtered_today_df = filtered_today_df[filtered_today_df['Current User Id'] == today_user_filter]
            
            # Today's data table
            st.subheader("📋 Today's Activities Details")
            
            if filtered_today_df.empty:
                st.info("No activities found today matching the current filters.")
            else:
                # Results count and download button
                results_col1, results_col2 = st.columns([3, 1])
                
                with results_col1:
                    st.markdown(f"**Total Today's Records:** {filtered_today_df.shape[0]}")
                
                with results_col2:
                    excel_data = generate_excel_download(filtered_today_df)
                    st.download_button(
                        label="📥 Download Today's Data",
                        data=excel_data,
                        file_name=f"today_activities_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                # Important columns for today's display
                today_display_cols = ['Case Id', 'Current User Id', 'Case Start Date', 'Type', 'Ticket Number', 'Status']
                
                # Add SR Status columns if available
                if 'SR Status' in filtered_today_df.columns:
                    today_display_cols.extend(['SR Status', 'Last Update'])
                
                # Ensure all columns exist
                today_display_cols = [col for col in today_display_cols if col in filtered_today_df.columns]
                
                # Display today's data
                st.dataframe(filtered_today_df[today_display_cols], hide_index=True)
                
                # Detailed today's case viewer
                st.subheader("🔍 Today's Case Details")
                
                selected_today_case = st.selectbox(
                    "Select a case to view details:",
                    filtered_today_df['Case Id'].tolist()
                )
                
                if selected_today_case:
                    today_row = filtered_today_df[filtered_today_df['Case Id'] == selected_today_case].iloc[0]
                    
                    # Display case details in a table
                    today_details = {
                        "Field": ["Case ID", "Owner", "Start Date", "Type", "Ticket Number", "Status"],
                        "Value": [
                            today_row['Case Id'],
                            today_row['Current User Id'],
                            today_row['Case Start Date'].strftime('%Y-%m-%d'),
                            today_row['Type'] if not pd.isna(today_row['Type']) else 'N/A',
                            int(today_row['Ticket Number']) if not pd.isna(today_row['Ticket Number']) else 'N/A',
                            today_row['Status']
                        ]
                    }
                    
                    # Add SR Status if available
                    if 'SR Status' in today_row and not pd.isna(today_row['SR Status']):
                        today_details["Field"].append("SR Status")
                        today_details["Value"].append(today_row['SR Status'])
                        
                        if 'Last Update' in today_row and not pd.isna(today_row['Last Update']):
                            today_details["Field"].append("Last Update")
                            today_details["Value"].append(today_row['Last Update'])
                    
                    # Display as a table
                    st.table(pd.DataFrame(today_details))
                    
                    # Display the full note
                    st.markdown("### Last Note")
                    if 'Last Note' in today_row and not pd.isna(today_row['Last Note']):
                        st.text_area("Note Content", today_row['Last Note'], height=200)
                    else:
                        st.info("No notes available for this case")
        else:
            st.warning("Today's data not available. Please ensure your main data includes 'Last Note Date' column.")

# Add footer
st.markdown("---")
st.markdown(
    """<div style="text-align:center; color:#888; font-size:0.8em;">
    SR Analyzer Pro v1.0 | Developed by Ali Babiker | © 2025
    </div>""",
    unsafe_allow_html=True
)