import streamlit as st
import os
import pandas as pd
import numpy as np
import re
import io
import base64
from datetime import datetime, timedelta
import pytz
from streamlit_option_menu import option_menu
import plotly.express as px
from utils import calculate_team_status_summary

# Set page configuration
st.set_page_config(
    page_title="Intellipen SmartQ",
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
if 'incident_df' not in st.session_state:
    st.session_state.incident_df = None
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = None
if 'last_upload_time' not in st.session_state:
    st.session_state.last_upload_time = None
if 'selected_users' not in st.session_state:
    st.session_state.selected_users = []
if 'selected_case_ids' not in st.session_state:
    st.session_state.selected_case_ids = []
if 'incident_overview_df' not in st.session_state:
    st.session_state.incident_overview_df = None
if 'report_datetime' not in st.session_state:
    st.session_state.report_datetime = None

@st.cache_data
def load_data(file):
    if file is None:
        return None, None  # Return None for both DataFrame and datetime string
    
    parsed_datetime_str = None
    df = None

    try:
        file_name = file.name
        # --- TEMPORARY LOGGING START ---
        print(f"--- DEBUG: load_data: file_name is '{file_name}' ---") 
        # --- TEMPORARY LOGGING END ---
        file_extension = os.path.splitext(file_name)[1].lower()

        # Attempt to parse datetime from filename
        match = re.search(r'_(\d{8})_(\d{6})\.', file_name)
        # --- TEMPORARY LOGGING START ---
        print(f"--- DEBUG: load_data: regex match object is {match} ---")
        if match:
            print(f"--- DEBUG: load_data: match.group(1) is '{match.group(1)}', match.group(2) is '{match.group(2)}' ---")
        # --- TEMPORARY LOGGING END ---
        
        if match:
            date_str = match.group(1)
            time_str = match.group(2)
            try:
                dt_object = datetime.strptime(date_str + time_str, '%Y%m%d%H%M%S')
                parsed_datetime_str = dt_object.strftime('%Y-%m-%d %H:%M:%S')
                # --- TEMPORARY LOGGING START ---
                print(f"--- DEBUG: load_data: successfully parsed datetime: {parsed_datetime_str} ---")
                # --- TEMPORARY LOGGING END ---
            except ValueError as e:
                # parsed_datetime_str remains None if parsing fails
                # --- TEMPORARY LOGGING START ---
                print(f"--- DEBUG: load_data: ValueError during parsing: {e} ---")
                # --- TEMPORARY LOGGING END ---
                # Optionally, log this error: st.warning(f"Could not parse date/time from filename: {file_name}")
        else:
            # --- TEMPORARY LOGGING START ---
            print(f"--- DEBUG: load_data: No regex match for filename '{file_name}' ---")
            # --- TEMPORARY LOGGING END ---
            # Optionally, log this error: st.warning(f"Could not parse date/time from filename: {file_name}")
        
        # Read the Excel file
        if file_extension == '.xls':
            try:
                df = pd.read_excel(file, engine='xlrd')
            except Exception:
                try:
                    file.seek(0)
                    df = pd.read_excel(file, engine='openpyxl')
                except Exception as e_openpyxl:
                    raise e_openpyxl
        elif file_extension == '.xlsx':
            df = pd.read_excel(file, engine='openpyxl')
        else:
            st.error(f"Unsupported file type: {file_extension}. Please upload .xls or .xlsx files.") # Changed from raise to st.error
            return None, parsed_datetime_str # Return None for df if unsupported, but retain parsed_datetime_str if filename was parsable

        return df, parsed_datetime_str
            
    except Exception as e:
        st.error(f"Error loading file '{file.name}': {e}")
        return None, parsed_datetime_str # Return None for df, but parsed_datetime_str might have a value if filename parsing happened before exception

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
        # SR numbers typically between 14000-18000 (adjust based on your system)
        ticket_type = "SR" if 14000 <= ticket_num <= 18000 else "Incident"
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
    # Display the logo
    st.image("Smart Q Logo.jpg", width=150)
    st.title("📊 Intellipen SmartQ")
    st.markdown("---")

    st.subheader("📁 Data Import")
    uploaded_file = st.file_uploader("Upload Main Excel File (.xlsx)", type=["xlsx"])
    sr_status_file = st.file_uploader("Upload SR Status Excel (optional)", type=["xlsx","xls"])
    incident_status_file = st.file_uploader("Upload Incident Report Excel (optional)", type=["xlsx","xls"])
    
    # report_datetime is initialized to None at the start of the session.
    # We process files in order: Main, SR, Incident for setting it IF it's currently None.

    if uploaded_file:
        with st.spinner("Loading main data..."):
            df, parsed_dt = load_data(uploaded_file)
            if df is not None:
                st.session_state.main_df = process_main_df(df)
                abu_dhabi_tz = pytz.timezone('Asia/Dubai')
                st.session_state.last_upload_time = datetime.now(abu_dhabi_tz).strftime("%Y-%m-%d %H:%M:%S")
                st.success(f"Main data loaded: {df.shape[0]} records")
                st.session_state.data_loaded = True
                if parsed_dt:
                    st.session_state.report_datetime = parsed_dt
            # If df is None, load_data would have shown an error. parsed_dt might still have a value if filename was parsable but content failed.
            # However, if df is None, we probably shouldn't use its parsed_dt either, as the file is unusable.
            # The current load_data returns (None, parsed_dt) even on file read error.
            # Consider if parsed_dt should only be used if df is not None.
            # For now, strictly following: if parsed_dt is not None from main file, use it.
            # This was already handled by: if parsed_dt: st.session_state.report_datetime = parsed_dt. This is fine.

    if sr_status_file:
        with st.spinner("Loading SR status data..."):
            sr_df, parsed_dt_sr = load_data(sr_status_file)
            if sr_df is not None:
                st.session_state.sr_df = sr_df
                st.success(f"SR status data loaded: {sr_df.shape[0]} records")
                if st.session_state.report_datetime is None and parsed_dt_sr:
                    st.session_state.report_datetime = parsed_dt_sr
            # else: df is None, error shown by load_data
    
    if incident_status_file:
        with st.spinner("Loading incident report data..."):
            incident_df, parsed_dt_incident = load_data(incident_status_file)
            if incident_df is not None:
                st.session_state.incident_df = incident_df
                st.success(f"Incident report data loaded: {incident_df.shape[0]} records")
                if st.session_state.report_datetime is None and parsed_dt_incident:
                    st.session_state.report_datetime = parsed_dt_incident

                # Process for Incident Overview tab (existing logic)
                overview_df = incident_df.copy()
                if "Customer" in overview_df.columns:
                    overview_df.rename(columns={"Customer": "Creator"}, inplace=True)
                st.session_state.incident_overview_df = overview_df
                st.success(f"Incident Overview data loaded: {len(overview_df)} records, {len(overview_df.columns)} columns.")
            else:
                st.session_state.incident_overview_df = None
    
    # Display last upload time (existing logic)
    if 'last_upload_time' not in st.session_state or st.session_state.last_upload_time is None:
        pass
    if st.session_state.get('last_upload_time'):
        st.info(f"Last data import: {st.session_state.last_upload_time}")
    else:
        st.info("No data imported yet in this session.")
    
    st.markdown("---")
    
    # Filters section (existing logic, depends on st.session_state.data_loaded)
    if st.session_state.data_loaded:
        st.subheader("🔍 Filters")
        df_main = st.session_state.main_df.copy() # Should be safe as data_loaded is True
        all_users = df_main['Current User Id'].dropna().unique().tolist()
        SELECT_ALL_USERS_OPTION = "[Select All Users]"
        default_users_hardcoded = ['ali.babiker', 'anas.hasan', 'ahmed.mostafa']
        default_users = [u for u in default_users_hardcoded if u in all_users]

        if 'sidebar_user_widget_selection_controlled' not in st.session_state:
            st.session_state.selected_users = list(default_users)
            if not default_users and all_users:
                st.session_state.selected_users = list(all_users)
                st.session_state.sidebar_user_widget_selection_controlled = [SELECT_ALL_USERS_OPTION]
            elif all_users and set(default_users) == set(all_users):
                st.session_state.sidebar_user_widget_selection_controlled = [SELECT_ALL_USERS_OPTION]
            elif not all_users:
                 st.session_state.selected_users = []
                 st.session_state.sidebar_user_widget_selection_controlled = []
            else:
                st.session_state.sidebar_user_widget_selection_controlled = list(default_users)

        options_for_user_widget = [SELECT_ALL_USERS_OPTION] + all_users
        raw_widget_selection = st.multiselect(
            "Select Users",
            options=options_for_user_widget,
            default=st.session_state.sidebar_user_widget_selection_controlled,
            key="multi_select_sidebar_users"
        )

        prev_widget_display_state = list(st.session_state.sidebar_user_widget_selection_controlled)
        current_select_all_option_selected = SELECT_ALL_USERS_OPTION in raw_widget_selection
        currently_selected_actual_items = [u for u in raw_widget_selection if u != SELECT_ALL_USERS_OPTION]
        user_clicked_select_all = current_select_all_option_selected and (SELECT_ALL_USERS_OPTION not in prev_widget_display_state)
        user_clicked_unselect_all = (not current_select_all_option_selected) and (SELECT_ALL_USERS_OPTION in prev_widget_display_state and len(prev_widget_display_state) == 1)

        if user_clicked_select_all:
            st.session_state.selected_users = list(all_users)
            st.session_state.sidebar_user_widget_selection_controlled = [SELECT_ALL_USERS_OPTION]
        elif user_clicked_unselect_all:
            st.session_state.selected_users = []
            st.session_state.sidebar_user_widget_selection_controlled = []
        else:
            if current_select_all_option_selected:
                if len(currently_selected_actual_items) < len(all_users):
                    st.session_state.selected_users = list(currently_selected_actual_items)
                    st.session_state.sidebar_user_widget_selection_controlled = list(currently_selected_actual_items)
                else:
                    st.session_state.selected_users = list(all_users)
                    st.session_state.sidebar_user_widget_selection_controlled = [SELECT_ALL_USERS_OPTION]
            else:
                st.session_state.selected_users = list(currently_selected_actual_items)
                if all_users and set(currently_selected_actual_items) == set(all_users):
                    st.session_state.sidebar_user_widget_selection_controlled = [SELECT_ALL_USERS_OPTION]
                else:
                    st.session_state.sidebar_user_widget_selection_controlled = list(currently_selected_actual_items)
        
        if 'Case Start Date' in df_main.columns:
            min_date = df_main['Case Start Date'].min().date()
            max_date = df_main['Case Start Date'].max().date()
            if 'sidebar_date_range_value' not in st.session_state:
                st.session_state.sidebar_date_range_value = (min_date, max_date)
            if st.button("Select Full Range", key="btn_select_full_date_range"):
                st.session_state.sidebar_date_range_value = (min_date, max_date)
            current_date_range_from_widget = st.date_input(
                "Date Range",
                value=st.session_state.sidebar_date_range_value,
                min_value=min_date,
                max_value=max_date,
                key="date_input_sidebar"
            )
            if current_date_range_from_widget != st.session_state.sidebar_date_range_value:
                st.session_state.sidebar_date_range_value = current_date_range_from_widget
            date_range = st.session_state.sidebar_date_range_value

# Main content
if not st.session_state.data_loaded:
    st.title("📊 Intellipen SmartQ")
    st.markdown("""
    ### Welcome to the Intellipen SmartQ!
    
    This application helps you analyze Service Requests and Incidents efficiently.
    
    To get started:
    1. Upload your main Excel file using the sidebar
    2. Optionally upload SR status file for enhanced analysis
    3. Optionally upload Incident report file for incident tracking
    4. Use the application to analyze and export your data
    
    **Enhanced Features:**
    - Advanced filtering and search
    - Detailed SR and Incident Analysis
    - SLA/Incident Breach monitoring
    - Today's new incidents and SRs
    - Unified Status tracking for both SRs and Incidents
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
        options=["Analysis", "SLA Breach", "Today's SR/Incidents", "Incident Overview"],
        icons=["kanban", "exclamation-triangle", "calendar-date", "clipboard-data"],
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
        df_enriched = df.copy()
        
        # Classify and extract ticket info
        if 'Last Note' in df_enriched.columns:
            df_enriched[['Triage Status', 'Ticket Number', 'Type']] = pd.DataFrame(
                df_enriched['Last Note'].apply(lambda x: pd.Series(classify_and_extract(x)))
            )
        else:
            df_enriched['Triage Status'] = "Error: Last Note missing"
            df_enriched['Ticket Number'] = None
            df_enriched['Type'] = None

        # Calculate case age
        if 'Case Start Date' in df_enriched.columns:
            df_enriched['Age (Days)'] = df_enriched['Case Start Date'].apply(calculate_age)
        else:
            df_enriched['Age (Days)'] = None

        # Determine if note was created today
        if 'Last Note Date' in df_enriched.columns:
            df_enriched['Created Today'] = df_enriched['Last Note Date'].apply(is_created_today)
        else:
            df_enriched['Created Today'] = False
        
        # Initialize Status, Last Update, and Breach Passed columns
        df_enriched['Status'] = None
        df_enriched['Last Update'] = None
        df_enriched['Breach Passed'] = None
        
        # Ensure 'Ticket Number' is numeric before any merges
        if 'Ticket Number' in df_enriched.columns:
            df_enriched['Ticket Number'] = pd.to_numeric(df_enriched['Ticket Number'], errors='coerce')

        # Merge with SR status data if available
        if hasattr(st.session_state, 'sr_df') and st.session_state.sr_df is not None:
            sr_df_copy = st.session_state.sr_df.copy()
            
            if 'Service Request' in sr_df_copy.columns:
                sr_df_copy['Service Request'] = sr_df_copy['Service Request'].astype(str).str.extract(r'(\d{4,})')
                sr_df_copy['Service Request'] = pd.to_numeric(sr_df_copy['Service Request'], errors='coerce')
                sr_df_copy.dropna(subset=['Service Request'], inplace=True)

                cols_to_merge_from_sr = ['Service Request']
                sr_rename_for_merge = {}

                if 'Status' in sr_df_copy.columns:
                    sr_rename_for_merge['Status'] = 'SR_Status_temp'
                if 'LastModDateTime' in sr_df_copy.columns:
                    sr_rename_for_merge['LastModDateTime'] = 'SR_Last_Update_temp'
                if 'Breach Passed' in sr_df_copy.columns:
                    sr_rename_for_merge['Breach Passed'] = 'SR_Breach_Value_temp'

                sr_df_copy.rename(columns=sr_rename_for_merge, inplace=True)

                for new_name in sr_rename_for_merge.values():
                    if new_name not in cols_to_merge_from_sr:
                        cols_to_merge_from_sr.append(new_name)

                df_enriched = df_enriched.merge(
                    sr_df_copy[cols_to_merge_from_sr],
                    how='left',
                    left_on='Ticket Number',
                    right_on='Service Request',
                    suffixes=('', '_sr_merged')
                )

                if 'Service Request_sr_merged' in df_enriched.columns:
                    df_enriched.drop(columns=['Service Request_sr_merged'], inplace=True)
                elif 'Service Request' in df_enriched.columns and 'Ticket Number' in df_enriched.columns and df_enriched.columns.tolist().count('Service Request') > 1:
                     df_enriched.drop(columns=['Service Request'], errors='ignore', inplace=True)

                sr_mask = df_enriched['Type'] == 'SR'

                if 'SR_Status_temp' in df_enriched.columns:
                    df_enriched.loc[sr_mask, 'Status'] = df_enriched.loc[sr_mask, 'SR_Status_temp']
                    df_enriched.drop(columns=['SR_Status_temp'], inplace=True)
                if 'SR_Last_Update_temp' in df_enriched.columns:
                    df_enriched.loc[sr_mask, 'Last Update'] = df_enriched.loc[sr_mask, 'SR_Last_Update_temp']
                    df_enriched.drop(columns=['SR_Last_Update_temp'], inplace=True)

                if 'SR_Breach_Value_temp' in df_enriched.columns:
                    def map_str_to_bool_sr(value):
                        if pd.isna(value): return None
                        val_lower = str(value).lower()
                        if val_lower in ['yes', 'true', '1', 'passed'] : return True
                        if val_lower in ['no', 'false', '0', 'failed']: return False
                        return None # Default for unmapped strings

                    mapped_values = df_enriched.loc[sr_mask, 'SR_Breach_Value_temp'].apply(map_str_to_bool_sr)
                    df_enriched.loc[sr_mask, 'Breach Passed'] = mapped_values
                    df_enriched.drop(columns=['SR_Breach_Value_temp'], inplace=True)

        # Merge with Incident status data if available
        if hasattr(st.session_state, 'incident_df') and st.session_state.incident_df is not None:
            incident_df_copy = st.session_state.incident_df.copy()
            incident_id_col_options = ['Incident', 'Incident ID', 'IncidentID', 'ID', 'Number']
            incident_id_col = None
            for col_option in incident_id_col_options:
                if col_option in incident_df_copy.columns:
                    incident_id_col = col_option
                    break
            
            if incident_id_col:
                incident_df_copy[incident_id_col] = incident_df_copy[incident_id_col].astype(str).str.extract(r'(\d{4,})')
                incident_df_copy[incident_id_col] = pd.to_numeric(incident_df_copy[incident_id_col], errors='coerce')
                incident_df_copy.dropna(subset=[incident_id_col], inplace=True)
                
                inc_rename_map = {incident_id_col: 'Incident_Number_temp'}
                inc_merge_cols = ['Incident_Number_temp']

                if 'Status' in incident_df_copy.columns:
                    inc_rename_map['Status'] = 'INC_Status_temp'
                    inc_merge_cols.append('INC_Status_temp')

                last_update_col_incident = None
                if 'Last Checked at' in incident_df_copy.columns: last_update_col_incident = 'Last Checked at'
                elif 'Last Checked atc' in incident_df_copy.columns: last_update_col_incident = 'Last Checked atc'
                elif 'Modified On' in incident_df_copy.columns: last_update_col_incident = 'Modified On'
                elif 'Last Update' in incident_df_copy.columns: last_update_col_incident = 'Last Update'

                if last_update_col_incident:
                    inc_rename_map[last_update_col_incident] = 'INC_Last_Update_temp'
                    inc_merge_cols.append('INC_Last_Update_temp')

                if 'Breach Passed' in incident_df_copy.columns:
                    inc_rename_map['Breach Passed'] = 'INC_Breach_Passed_temp'
                    inc_merge_cols.append('INC_Breach_Passed_temp')

                incident_df_copy.rename(columns=inc_rename_map, inplace=True)
                
                df_enriched = df_enriched.merge(
                    incident_df_copy[inc_merge_cols],
                    how='left',
                    left_on='Ticket Number',
                    right_on='Incident_Number_temp',
                    suffixes=('', '_inc_merged')
                )
                if 'Incident_Number_temp_inc_merged' in df_enriched.columns:
                     df_enriched.drop(columns=['Incident_Number_temp_inc_merged'], inplace=True)
                elif 'Incident_Number_temp' in df_enriched.columns :
                     df_enriched.drop(columns=['Incident_Number_temp'], inplace=True, errors='ignore')

                incident_mask = df_enriched['Type'] == 'Incident'
                
                if 'INC_Status_temp' in df_enriched.columns:
                    df_enriched.loc[incident_mask, 'Status'] = df_enriched.loc[incident_mask, 'INC_Status_temp']
                    df_enriched.drop(columns=['INC_Status_temp'], inplace=True)
                if 'INC_Last_Update_temp' in df_enriched.columns:
                    df_enriched.loc[incident_mask, 'Last Update'] = df_enriched.loc[incident_mask, 'INC_Last_Update_temp']
                    df_enriched.drop(columns=['INC_Last_Update_temp'], inplace=True)
                
                if 'INC_Breach_Passed_temp' in df_enriched.columns:
                    def map_str_to_bool_inc(value):
                        if pd.isna(value): return None
                        if isinstance(value, bool): return value
                        val_lower = str(value).lower()
                        if val_lower in ['yes', 'true', '1', 'passed', 'breached']: return True
                        if val_lower in ['no', 'false', '0', 'failed', 'not breached']: return False
                        return None

                    mapped_inc_breach_values = df_enriched.loc[incident_mask, 'INC_Breach_Passed_temp'].apply(map_str_to_bool_inc)
                    df_enriched.loc[incident_mask, 'Breach Passed'] = mapped_inc_breach_values
                    df_enriched.drop(columns=['INC_Breach_Passed_temp'], inplace=True)

        if 'Last Update' in df_enriched.columns:
            df_enriched['Last Update'] = pd.to_datetime(df_enriched['Last Update'], errors='coerce')
        if 'Breach Date' in df_enriched.columns:
            df_enriched['Breach Date'] = pd.to_datetime(df_enriched['Breach Date'], errors='coerce')

        if 'Ticket Number' in df_enriched.columns and 'Type' in df_enriched.columns:
            valid_ticket_mask = df_enriched['Ticket Number'].notna() & df_enriched['Type'].notna()
            if valid_ticket_mask.any():
                 df_enriched.loc[valid_ticket_mask, 'Case Count'] = df_enriched[valid_ticket_mask].groupby(['Ticket Number', 'Type'])['Ticket Number'].transform('size')
        else:
            df_enriched['Case Count'] = pd.NA
            
        return df_enriched
    
    # Enrich data with classifications and metrics
    df_enriched = enrich_data(df_filtered)
    
    # Store the enriched dataframe for use across tabs
    st.session_state.filtered_df = df_enriched
    
    #
    # ANALYSIS TAB (Formerly SR/INCIDENT ANALYSIS)
    #
    if selected == "Analysis":
        st.title("🔍 Analysis")
        
        # Adjusted "Last data update" display
        if st.session_state.get('last_upload_time'):
            # Use a clearer label now that last_upload_time is corrected
            update_time_display = st.session_state.last_upload_time
            st.markdown(f"**Last Data Import Time:** {update_time_display}")
        else:
            st.markdown("**Last Data Import Time:** No data imported yet")

        # New "Report Datetime" label
        if st.session_state.get('report_datetime'):
            report_datetime_display = st.session_state.report_datetime
            st.markdown(f"**Report Datetime (from filename):** {report_datetime_display}")
        else:
            st.markdown("**Report Datetime (from filename):** Not available")
        
        # Filtering options
        col1, col2,col3 = st.columns(3)
        
        with col1:
            status_filter = st.selectbox(
                "Filter by Triage Status",
                ["All"] + df_enriched["Triage Status"].dropna().unique().tolist()
            )
        
        with col2:
            type_filter = st.selectbox(
                "Filter by Type",
                ["All", "SR", "Incident"]
            )
        
        with col3:
            # Unified Status filter
            if 'Status' in df_enriched.columns:
                status_options = ["All"] + df_enriched['Status'].dropna().unique().tolist() + ["None"]
                unified_status_filter = st.selectbox("Filter by Status", status_options)
            else:
                unified_status_filter = "All"
        
        # Apply filters
        df_display = df_enriched.copy()
        
        if status_filter != "All":
            df_display = df_display[df_display["Triage Status"] == status_filter]
        
        if type_filter != "All":
            df_display = df_display[df_display["Type"] == type_filter]
        
        if unified_status_filter != "All":
            if unified_status_filter == "None":
                df_display = df_display[df_display["Status"].isna()]
            else:
                df_display = df_display[df_display["Status"] == unified_status_filter]
        
        # Statistics and summary
        st.subheader("📊 Summary Analysis")
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.markdown("**🔸 Triage Status Count**")
            triage_summary = df_enriched['Triage Status'].value_counts().rename_axis('Triage Status').reset_index(name='Count')
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
        summary_col3, summary_col4 = st.columns(2)
        with summary_col3:
            st.markdown("**🟢 SR Status Summary**")
            if 'Status' in df_enriched.columns and 'Type' in df_enriched.columns and not df_enriched[df_enriched['Type'] == 'SR'].empty:
                # Filter only for SRs
                df_srs = df_enriched[df_enriched['Type'] == 'SR'].copy()
                df_srs_status_valid = df_srs.dropna(subset=['Status'])
                
                if not df_srs_status_valid.empty:
                    # All SR status count
                    status_all_counts = df_srs_status_valid['Status'].value_counts().rename_axis('Status').reset_index(name='All Count')
                    
                    # Unique SR tickets
                    ticket_unique = df_srs_status_valid.dropna(subset=['Ticket Number'])[['Ticket Number', 'Status']].drop_duplicates()
                    ticket_unique_counts = ticket_unique['Status'].value_counts().rename_axis('Status').reset_index(name='Unique Count')
                    
                    # Merge both summaries
                    merged_status = pd.merge(status_all_counts, ticket_unique_counts, on='Status', how='outer').fillna(0)
                    merged_status[['All Count', 'Unique Count']] = merged_status[['All Count', 'Unique Count']].astype(int)
                    
                    # Total row
                    total_row = {
                        'Status': 'Total',
                        'All Count': merged_status['All Count'].sum(),
                        'Unique Count': merged_status['Unique Count'].sum()
                    }
                    
                    status_summary_df = pd.concat([merged_status, pd.DataFrame([total_row])], ignore_index=True)
                    
                    # Display
                    st.dataframe(
                        status_summary_df.style.apply(
                            lambda x: ['background-color: #bbdefb; font-weight: bold' if x.name == len(status_summary_df)-1 else '' for _ in x],
                            axis=1
                        )
                    )
                else:
                    st.info("No SRs with status information available.")
            else:
                st.info("Upload SR Status Excel file to view SR Status Summary.")

        #Incident Status Summary
        with summary_col4: # Or create new columns if layout needs adjustment
            st.markdown("**🟣 Incident Status Summary**")
            if 'Status' in df_enriched.columns and 'Type' in df_enriched.columns and not df_enriched[df_enriched['Type'] == 'Incident'].empty:
                df_incidents = df_enriched[df_enriched['Type'] == 'Incident'].copy()
                df_incidents_status_valid = df_incidents.dropna(subset=['Status'])

                if not df_incidents_status_valid.empty:
                    # All incident status count
                    incident_status_all_counts = df_incidents_status_valid['Status'].value_counts().rename_axis('Status').reset_index(name='Cases Count')
                    
                    # Unique incident tickets
                    incident_ticket_unique = df_incidents_status_valid.dropna(subset=['Ticket Number'])[['Ticket Number', 'Status']].drop_duplicates()
                    incident_ticket_unique_counts = incident_ticket_unique['Status'].value_counts().rename_axis('Status').reset_index(name='Unique Count')
                    
                    # Merge both summaries for incidents
                    merged_incident_status = pd.merge(incident_status_all_counts, incident_ticket_unique_counts, on='Status', how='outer').fillna(0)
                    merged_incident_status[['Cases Count', 'Unique Count']] = merged_incident_status[['Cases Count', 'Unique Count']].astype(int)
                    
                    # Total row for incidents
                    incident_total_row = {
                        'Status': 'Total',
                        'Cases Count': merged_incident_status['Cases Count'].sum(),
                        'Unique Count': merged_incident_status['Unique Count'].sum()
                    }
                    
                    incident_status_summary_df = pd.concat([merged_incident_status, pd.DataFrame([incident_total_row])], ignore_index=True)
                    
                    # Display Incident Status Summary
                    st.dataframe(
                        incident_status_summary_df.style.apply(
                            lambda x: ['background-color: #bbdefb; font-weight: bold' if x.name == len(incident_status_summary_df)-1 else '' for _ in x],
                            axis=1
                        )
                    )
                else:
                    st.info("No incidents with status information available to summarize.")
            elif st.session_state.incident_df is None:
                st.info("Upload Incident Report Excel file to view Incident Status Summary.")
            else:
                st.info("No incident data available to summarize.")
        
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
                    file_name=f"sr_incident_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        # Display data table with customizable columns
        if not df_display.empty:
            all_columns = df_display.columns.tolist()
            SELECT_ALL_COLS_ANALYSIS_OPTION = "[Select All Columns]"

            # Define default columns (original logic)
            default_selected_cols_initial = ['Last Note', 'Case Id', 'Current User Id', 'Case Start Date', 'Triage Status', 'Type', 'Ticket Number']
            if 'Status' in df_display.columns:
                default_selected_cols_initial.extend(['Status', 'Last Update'])
            if 'Breach Passed' in df_display.columns:
                default_selected_cols_initial.append('Breach Passed')

            # Ensure default columns are valid and exist in df_display
            default_selected_cols = [col for col in default_selected_cols_initial if col in all_columns]

            if 'analysis_tab_column_widget_selection_controlled' not in st.session_state:
                st.session_state.selected_display_cols = list(default_selected_cols) # Actual columns to display

                if not default_selected_cols and all_columns: # If default_selected_cols is empty and there are columns, select all
                    st.session_state.selected_display_cols = list(all_columns)
                    st.session_state.analysis_tab_column_widget_selection_controlled = [SELECT_ALL_COLS_ANALYSIS_OPTION]
                elif all_columns and set(default_selected_cols) == set(all_columns): # If default_selected_cols happens to be all columns
                    st.session_state.analysis_tab_column_widget_selection_controlled = [SELECT_ALL_COLS_ANALYSIS_OPTION]
                elif not all_columns: # If there are no columns at all
                    st.session_state.selected_display_cols = []
                    st.session_state.analysis_tab_column_widget_selection_controlled = []
                else: # Default columns are a subset
                    st.session_state.analysis_tab_column_widget_selection_controlled = list(default_selected_cols)

            options_for_cols_widget = [SELECT_ALL_COLS_ANALYSIS_OPTION] + all_columns
            raw_cols_widget_selection = st.multiselect(
                "Select columns to display:",
                options=options_for_cols_widget,
                default=st.session_state.analysis_tab_column_widget_selection_controlled,
                key="multi_select_analysis_columns"
            )

            prev_widget_display_state = list(st.session_state.analysis_tab_column_widget_selection_controlled)
            current_select_all_option_selected = SELECT_ALL_COLS_ANALYSIS_OPTION in raw_cols_widget_selection
            currently_selected_actual_items = [c for c in raw_cols_widget_selection if c != SELECT_ALL_COLS_ANALYSIS_OPTION]

            user_clicked_select_all = current_select_all_option_selected and (SELECT_ALL_COLS_ANALYSIS_OPTION not in prev_widget_display_state)
            user_clicked_unselect_all = (not current_select_all_option_selected) and (SELECT_ALL_COLS_ANALYSIS_OPTION in prev_widget_display_state and len(prev_widget_display_state) == 1)

            if user_clicked_select_all:
                st.session_state.selected_display_cols = list(all_columns)
                st.session_state.analysis_tab_column_widget_selection_controlled = [SELECT_ALL_COLS_ANALYSIS_OPTION]
            elif user_clicked_unselect_all:
                st.session_state.selected_display_cols = []
                st.session_state.analysis_tab_column_widget_selection_controlled = []
            else:
                if current_select_all_option_selected:
                    if len(currently_selected_actual_items) < len(all_columns):
                        st.session_state.selected_display_cols = list(currently_selected_actual_items)
                        st.session_state.analysis_tab_column_widget_selection_controlled = list(currently_selected_actual_items)
                    else:
                        st.session_state.selected_display_cols = list(all_columns)
                        st.session_state.analysis_tab_column_widget_selection_controlled = [SELECT_ALL_COLS_ANALYSIS_OPTION]
                else:
                    st.session_state.selected_display_cols = list(currently_selected_actual_items)
                    if all_columns and set(currently_selected_actual_items) == set(all_columns):
                        st.session_state.analysis_tab_column_widget_selection_controlled = [SELECT_ALL_COLS_ANALYSIS_OPTION]
                    else:
                        st.session_state.analysis_tab_column_widget_selection_controlled = list(currently_selected_actual_items)

            columns_to_show = st.session_state.get('selected_display_cols', []) # Default to empty list if not set
            if not columns_to_show and all_columns:
                 columns_to_show = all_columns

            if columns_to_show: # Ensure there are columns to show
                st.dataframe(df_display[columns_to_show], hide_index=True)
            else: # This case should ideally be covered by the logic above, but as a fallback
                st.info("Please select at least one column to display, or all columns will be shown if the selection is empty and columns are available.")

        elif df_display.empty: # This elif should be at the same level as the first if not df_display.empty
            st.info("No data to display based on current filters.")
        # If df_display is not empty but columns_to_show ended up empty (e.g. if all_columns was also empty initially)
        # This case is unlikely if df_display was not empty, but good to be robust.
        else:
            st.info("No columns available to display.")

        # Incidents/SRs Linked Cases Summary
        st.subheader("🔗 Incidents/SRs Linked Cases Summary")
        min_linked_cases = st.number_input("Minimum Linked Cases", min_value=1, value=2, step=1)

        if 'Case Count' in df_display.columns and 'Ticket Number' in df_display.columns:
            linked_cases_df = df_display[
                (df_display['Case Count'] >= min_linked_cases) &
                (df_display['Ticket Number'].notna())
            ]

            if not linked_cases_df.empty:
                linked_summary_df = linked_cases_df[['Ticket Number', 'Type','Status', 'Case Count']].drop_duplicates().sort_values(by='Case Count', ascending=False)
                st.dataframe(linked_summary_df, hide_index=True)
            else:
                st.info(f"No Incidents/SRs found with at least {min_linked_cases} linked cases based on current filters.")
        else:
            st.warning("Required columns ('Case Count', 'Ticket Number') not available for linked cases summary.")

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
            
            # Add Status if available
            if 'Status' in case_row and not pd.isna(case_row['Status']):
                case_details["Field"].append("Status")
                case_details["Value"].append(case_row['Status'])
                
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
        
        # Check if either SR or Incident data is available
        if st.session_state.sr_df is None and st.session_state.incident_df is None:
            st.warning("Please upload SR Status Excel file or Incident Report Excel file to view SLA breach information.")
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
                    if 'Status' in breach_df.columns:
                        open_breaches = len(breach_df[breach_df['Status'].isin(['Open', 'In Progress', 'Pending', 'New'])])
                        st.markdown(f'<p class="metric-value">{open_breaches}</p>', unsafe_allow_html=True)
                        st.markdown('<p class="metric-label">Open Breached Cases</p>', unsafe_allow_html=True)
                    else:
                        st.markdown('<p class="metric-value">N/A</p>', unsafe_allow_html=True)
                        st.markdown('<p class="metric-label">Status Not Available</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    if 'Type' in breach_df.columns:
                        sr_breaches = len(breach_df[breach_df['Type'] == 'SR'])
                        st.markdown(f'<p class="metric-value">{sr_breaches}</p>', unsafe_allow_html=True)
                        st.markdown('<p class="metric-label">SR Breaches</p>', unsafe_allow_html=True)
                    else:
                        st.markdown('<p class="metric-value">N/A</p>', unsafe_allow_html=True)
                        st.markdown('<p class="metric-label">SR Breaches</p>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Breach details by type and status
                if not breach_df.empty:
                    st.subheader("📋 SLA Breach Details")
                    
                    # Filter options for breach analysis
                    breach_col1, breach_col2 = st.columns(2)
                    
                    with breach_col1:
                        breach_type_filter = st.selectbox(
                            "Filter by Type (Breach)",
                            ["All", "SR", "Incident"],
                            key="breach_type"
                        )
                    
                    with breach_col2:
                        if 'Status' in breach_df.columns:
                            breach_status_options = ["All"] + breach_df['Status'].dropna().unique().tolist()
                            breach_status_filter = st.selectbox(
                                "Filter by Status (Breach)",
                                breach_status_options,
                                key="breach_status"
                            )
                        else:
                            breach_status_filter = "All"
                    
                    # Apply breach filters
                    breach_display = breach_df.copy()
                    
                    if breach_type_filter != "All":
                        breach_display = breach_display[breach_display["Type"] == breach_type_filter]
                    
                    if breach_status_filter != "All":
                        breach_display = breach_display[breach_display["Status"] == breach_status_filter]
                    
                    # Display breach results
                    st.markdown(f"**Total Breached Records:** {breach_display.shape[0]}")
                    
                    # Download button for breach data
                    if not breach_display.empty:
                        excel_breach_data = generate_excel_download(breach_display)
                        st.download_button(
                            label="📥 Download Breach Analysis",
                            data=excel_breach_data,
                            file_name=f"sla_breach_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    # Breach data table
                    breach_cols = ['Case Id', 'Current User Id', 'Case Start Date', 'Type', 'Ticket Number', 'Status', 'Last Update', 'Age (Days)']
                    breach_display_cols = [col for col in breach_cols if col in breach_display.columns]
                    
                    if not breach_display.empty:
                        st.dataframe(breach_display[breach_display_cols], hide_index=True)
                    else:
                        st.info("No breached cases match the selected filters.")
                        
                else:
                    st.info("No SLA breaches found in the current dataset.")
            else:
                st.info("SLA breach information not available. Please ensure your SR/Incident status files contain 'Breach Passed' column.")
    
    #
    # TODAY'S SR/INCIDENTS TAB
    #
    elif selected == "Today's SR/Incidents":
        st.title("📅 Today's New SR/Incidents")
        
        # Get today's cases
        today = datetime.now().date()
        
        # Filter for cases with notes created today
        if 'Created Today' in df_enriched.columns:
            today_cases = df_enriched[df_enriched['Created Today'] == True].copy()
        else:
            # Fallback: filter by Last Note Date
            today_cases = df_enriched[df_enriched['Last Note Date'].dt.date == today].copy() if 'Last Note Date' in df_enriched.columns else pd.DataFrame()
        
        # Further filter for SR/Incident cases only
        today_sr_incidents = today_cases[today_cases['Triage Status'] == 'Pending SR/Incident'].copy()
        
        # Display summary
        st.subheader("📊 Today's Summary")
        
        summary_today_col1, summary_today_col2, summary_today_col3 = st.columns(3)
        
        with summary_today_col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            total_today = len(today_sr_incidents)
            st.markdown(f'<p class="metric-value">{total_today}</p>', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">Total New SR/Incidents</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with summary_today_col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if 'Type' in today_sr_incidents.columns:
                sr_today = len(today_sr_incidents[today_sr_incidents['Type'] == 'SR'])
                st.markdown(f'<p class="metric-value">{sr_today}</p>', unsafe_allow_html=True)
                st.markdown('<p class="metric-label">New SRs Today</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="metric-value">N/A</p>', unsafe_allow_html=True)
                st.markdown('<p class="metric-label">New SRs Today</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with summary_today_col3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if 'Type' in today_sr_incidents.columns:
                incident_today = len(today_sr_incidents[today_sr_incidents['Type'] == 'Incident'])
                st.markdown(f'<p class="metric-value">{incident_today}</p>', unsafe_allow_html=True)
                st.markdown('<p class="metric-label">New Incidents Today</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="metric-value">N/A</p>', unsafe_allow_html=True)
                st.markdown('<p class="metric-label">New Incidents Today</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Breakdown by user
        if not today_sr_incidents.empty:
            st.subheader("👥 Breakdown by User")
            
            user_breakdown = today_sr_incidents.groupby('Current User Id').agg({
                'Case Id': 'count',
                'Type': lambda x: (x == 'SR').sum(),
                'Ticket Number': lambda x: (today_sr_incidents.loc[x.index, 'Type'] == 'Incident').sum()
            }).rename(columns={
                'Case Id': 'Total',
                'Type': 'SRs',
                'Ticket Number': 'Incidents'
            })
            
            user_breakdown = user_breakdown.reset_index()
            
            # Add total row
            total_row = pd.DataFrame({
                'Current User Id': ['TOTAL'],
                'Total': [user_breakdown['Total'].sum()],
                'SRs': [user_breakdown['SRs'].sum()],
                'Incidents': [user_breakdown['Incidents'].sum()]
            })
            
            user_breakdown_display = pd.concat([user_breakdown, total_row], ignore_index=True)
            
            st.dataframe(
                user_breakdown_display.style.apply(
                    lambda x: ['background-color: #bbdefb; font-weight: bold' if x.name == len(user_breakdown_display)-1 else '' for _ in x],
                    axis=1
                )
            )
            
            # Filter options for today's data
            st.subheader("🔍 Filter Today's Data")
            
            today_col1, today_col2 = st.columns(2)
            
            with today_col1:
                today_user_filter = st.selectbox(
                    "Filter by User (Today)",
                    ["All"] + today_sr_incidents['Current User Id'].unique().tolist(),
                    key="today_user"
                )
            
            with today_col2:
                today_type_filter = st.selectbox(
                    "Filter by Type (Today)",
                    ["All", "SR", "Incident"],
                    key="today_type"
                )
            
            # Apply today's filters
            today_display = today_sr_incidents.copy()
            
            if today_user_filter != "All":
                today_display = today_display[today_display["Current User Id"] == today_user_filter]
            
            if today_type_filter != "All":
                today_display = today_display[today_display["Type"] == today_type_filter]
            
            # Display today's results
            st.subheader("📋 Today's Details")
            
            results_today_col1, results_today_col2 = st.columns([3, 1])
            
            with results_today_col1:
                st.markdown(f"**Filtered Records:** {today_display.shape[0]}")
            
            with results_today_col2:
                if not today_display.empty:
                    excel_today_data = generate_excel_download(today_display)
                    st.download_button(
                        label="📥 Download Today's Data",
                        data=excel_today_data,
                        file_name=f"todays_sr_incidents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            # Today's data table
            today_cols = ['Case Id', 'Current User Id', 'Last Note Date', 'Type', 'Ticket Number']
            if 'Status' in today_display.columns:
                today_cols.extend(['Status', 'Last Update'])
            
            today_display_cols = [col for col in today_cols if col in today_display.columns]
            
            if not today_display.empty:
                st.dataframe(today_display[today_display_cols], hide_index=True)
            else:
                st.info("No records match the selected filters for today.")
                
        else:
            st.info("No new SR/Incidents found for today.")
            
            # Show all today's cases (not just SR/Incidents)
            if not today_cases.empty:
                st.subheader("📝 All Today's Cases")
                st.markdown(f"**Total cases with notes today:** {len(today_cases)}")
                
                all_today_cols = ['Case Id', 'Current User Id', 'Last Note Date', 'Triage Status']
                all_today_display_cols = [col for col in all_today_cols if col in today_cases.columns]
                
                st.dataframe(today_cases[all_today_display_cols], hide_index=True)
                
                # Download button for all today's cases
                excel_all_today_data = generate_excel_download(today_cases)
                st.download_button(
                    label="📥 Download All Today's Cases",
                    data=excel_all_today_data,
                    file_name=f"all_todays_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.info("No cases found with notes created today.")

    elif selected == "Incident Overview":
        st.title("📋 Incident Overview")

        SELECT_ALL_BASE_STRING = "[Select All %s]"

        if 'incident_overview_df' not in st.session_state or st.session_state.incident_overview_df is None or st.session_state.incident_overview_df.empty:
            st.warning(
                "The 'Incident Report Excel' has not been uploaded or is missing the required columns "
                "('Customer', 'Incident', 'Team', 'Priority', 'Status'). " # Updated message
                "Please upload the correct file via the sidebar to view the Incident Overview."
            )
        else:
            overview_df = st.session_state.incident_overview_df.copy() # Work with a copy

            st.subheader("Filter Incidents")
            # Create 4 columns for filters to include Status
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                # Ensure 'Creator' column exists before trying to access it
                if 'Creator' in overview_df.columns:
                    unique_creators = sorted(overview_df['Creator'].dropna().unique())
                else:
                    unique_creators = [] # Default to empty list if column is missing

                SELECT_ALL_CREATORS_OPTION = SELECT_ALL_BASE_STRING % "Creators"

                if 'incident_creator_widget_selection_controlled' not in st.session_state:
                    st.session_state.selected_creators = list(unique_creators) # Default to all selected for filtering
                    if unique_creators: # If there are actual creators
                        st.session_state.incident_creator_widget_selection_controlled = [SELECT_ALL_CREATORS_OPTION]
                    else: # No creators
                        st.session_state.incident_creator_widget_selection_controlled = []

                options_for_creator_widget = [SELECT_ALL_CREATORS_OPTION] + unique_creators
                raw_creator_widget_selection = st.multiselect(
                    "Filter by Creator",
                    options=options_for_creator_widget,
                    default=st.session_state.incident_creator_widget_selection_controlled,
                    key="multi_select_incident_creator"
                )

                prev_widget_display_state = list(st.session_state.incident_creator_widget_selection_controlled)
                current_select_all_option_selected = SELECT_ALL_CREATORS_OPTION in raw_creator_widget_selection
                currently_selected_actual_items = [c for c in raw_creator_widget_selection if c != SELECT_ALL_CREATORS_OPTION]

                user_clicked_select_all = current_select_all_option_selected and (SELECT_ALL_CREATORS_OPTION not in prev_widget_display_state)
                user_clicked_unselect_all = (not current_select_all_option_selected) and (SELECT_ALL_CREATORS_OPTION in prev_widget_display_state and len(prev_widget_display_state) == 1)

                if user_clicked_select_all:
                    st.session_state.selected_creators = list(unique_creators)
                    st.session_state.incident_creator_widget_selection_controlled = [SELECT_ALL_CREATORS_OPTION]
                elif user_clicked_unselect_all:
                    st.session_state.selected_creators = []
                    st.session_state.incident_creator_widget_selection_controlled = []
                else:
                    if current_select_all_option_selected:
                        if len(currently_selected_actual_items) < len(unique_creators):
                            st.session_state.selected_creators = list(currently_selected_actual_items)
                            st.session_state.incident_creator_widget_selection_controlled = list(currently_selected_actual_items)
                        else:
                            st.session_state.selected_creators = list(unique_creators)
                            st.session_state.incident_creator_widget_selection_controlled = [SELECT_ALL_CREATORS_OPTION]
                    else:
                        st.session_state.selected_creators = list(currently_selected_actual_items)
                        if unique_creators and set(currently_selected_actual_items) == set(unique_creators):
                            st.session_state.incident_creator_widget_selection_controlled = [SELECT_ALL_CREATORS_OPTION]
                        else:
                            st.session_state.incident_creator_widget_selection_controlled = list(currently_selected_actual_items)

            with col2:
                if 'Team' in overview_df.columns:
                    unique_teams = sorted(overview_df['Team'].dropna().unique())
                else:
                    unique_teams = []

                SELECT_ALL_TEAMS_OPTION = SELECT_ALL_BASE_STRING % "Teams"

                if 'incident_team_widget_selection_controlled' not in st.session_state:
                    st.session_state.selected_teams = list(unique_teams) # Default to all selected for filtering
                    if unique_teams: # If there are actual teams
                        st.session_state.incident_team_widget_selection_controlled = [SELECT_ALL_TEAMS_OPTION]
                    else: # No teams
                        st.session_state.incident_team_widget_selection_controlled = []

                options_for_team_widget = [SELECT_ALL_TEAMS_OPTION] + unique_teams
                raw_team_widget_selection = st.multiselect(
                    "Filter by Team",
                    options=options_for_team_widget,
                    default=st.session_state.incident_team_widget_selection_controlled,
                    key="multi_select_incident_team"
                )

                prev_widget_display_state = list(st.session_state.incident_team_widget_selection_controlled)
                current_select_all_option_selected = SELECT_ALL_TEAMS_OPTION in raw_team_widget_selection
                currently_selected_actual_items = [t for t in raw_team_widget_selection if t != SELECT_ALL_TEAMS_OPTION]

                user_clicked_select_all = current_select_all_option_selected and (SELECT_ALL_TEAMS_OPTION not in prev_widget_display_state)
                user_clicked_unselect_all = (not current_select_all_option_selected) and (SELECT_ALL_TEAMS_OPTION in prev_widget_display_state and len(prev_widget_display_state) == 1)

                if user_clicked_select_all:
                    st.session_state.selected_teams = list(unique_teams)
                    st.session_state.incident_team_widget_selection_controlled = [SELECT_ALL_TEAMS_OPTION]
                elif user_clicked_unselect_all:
                    st.session_state.selected_teams = []
                    st.session_state.incident_team_widget_selection_controlled = []
                else:
                    if current_select_all_option_selected:
                        if len(currently_selected_actual_items) < len(unique_teams):
                            st.session_state.selected_teams = list(currently_selected_actual_items)
                            st.session_state.incident_team_widget_selection_controlled = list(currently_selected_actual_items)
                        else:
                            st.session_state.selected_teams = list(unique_teams)
                            st.session_state.incident_team_widget_selection_controlled = [SELECT_ALL_TEAMS_OPTION]
                    else:
                        st.session_state.selected_teams = list(currently_selected_actual_items)
                        if unique_teams and set(currently_selected_actual_items) == set(unique_teams):
                            st.session_state.incident_team_widget_selection_controlled = [SELECT_ALL_TEAMS_OPTION]
                        else:
                            st.session_state.incident_team_widget_selection_controlled = list(currently_selected_actual_items)

            with col3:
                if 'Priority' in overview_df.columns:
                    unique_priorities = sorted(overview_df['Priority'].dropna().unique())
                else:
                    unique_priorities = []

                SELECT_ALL_PRIORITIES_OPTION = SELECT_ALL_BASE_STRING % "Priorities"

                if 'incident_priority_widget_selection_controlled' not in st.session_state:
                    st.session_state.selected_priorities = list(unique_priorities) # Default to all selected for filtering
                    if unique_priorities: # If there are actual priorities
                        st.session_state.incident_priority_widget_selection_controlled = [SELECT_ALL_PRIORITIES_OPTION]
                    else: # No priorities
                        st.session_state.incident_priority_widget_selection_controlled = []

                options_for_priority_widget = [SELECT_ALL_PRIORITIES_OPTION] + unique_priorities
                raw_priority_widget_selection = st.multiselect(
                    "Filter by Priority",
                    options=options_for_priority_widget,
                    default=st.session_state.incident_priority_widget_selection_controlled,
                    key="multi_select_incident_priority"
                )

                prev_widget_display_state = list(st.session_state.incident_priority_widget_selection_controlled)
                current_select_all_option_selected = SELECT_ALL_PRIORITIES_OPTION in raw_priority_widget_selection
                currently_selected_actual_items = [p for p in raw_priority_widget_selection if p != SELECT_ALL_PRIORITIES_OPTION]

                user_clicked_select_all = current_select_all_option_selected and (SELECT_ALL_PRIORITIES_OPTION not in prev_widget_display_state)
                user_clicked_unselect_all = (not current_select_all_option_selected) and (SELECT_ALL_PRIORITIES_OPTION in prev_widget_display_state and len(prev_widget_display_state) == 1)

                if user_clicked_select_all:
                    st.session_state.selected_priorities = list(unique_priorities)
                    st.session_state.incident_priority_widget_selection_controlled = [SELECT_ALL_PRIORITIES_OPTION]
                elif user_clicked_unselect_all:
                    st.session_state.selected_priorities = []
                    st.session_state.incident_priority_widget_selection_controlled = []
                else:
                    if current_select_all_option_selected:
                        if len(currently_selected_actual_items) < len(unique_priorities):
                            st.session_state.selected_priorities = list(currently_selected_actual_items)
                            st.session_state.incident_priority_widget_selection_controlled = list(currently_selected_actual_items)
                        else:
                            st.session_state.selected_priorities = list(unique_priorities)
                            st.session_state.incident_priority_widget_selection_controlled = [SELECT_ALL_PRIORITIES_OPTION]
                    else:
                        st.session_state.selected_priorities = list(currently_selected_actual_items)
                        if unique_priorities and set(currently_selected_actual_items) == set(unique_priorities):
                            st.session_state.incident_priority_widget_selection_controlled = [SELECT_ALL_PRIORITIES_OPTION]
                        else:
                            st.session_state.incident_priority_widget_selection_controlled = list(currently_selected_actual_items)

            with col4: # New column for Status filter
                if 'Status' in overview_df.columns:
                    unique_statuses = sorted(overview_df['Status'].dropna().unique())
                    # Exclude 'Closed', 'Resolved', 'Cancelled' by default
                    closed_like_statuses = {'Closed', 'Cancelled'}
                    default_selected_statuses = [s for s in unique_statuses if s not in closed_like_statuses]
                else:
                    unique_statuses = []
                    default_selected_statuses = []

                SELECT_ALL_STATUSES_OPTION = SELECT_ALL_BASE_STRING % "Statuses"

                if 'incident_status_widget_selection_controlled' not in st.session_state:
                    st.session_state.selected_statuses = list(default_selected_statuses) # Actual statuses for filtering

                    if not default_selected_statuses and unique_statuses: # If default_selected_statuses is empty and there are statuses, select all
                        st.session_state.selected_statuses = list(unique_statuses)
                        st.session_state.incident_status_widget_selection_controlled = [SELECT_ALL_STATUSES_OPTION]
                    elif unique_statuses and set(default_selected_statuses) == set(unique_statuses): # If default_selected_statuses happens to be all unique_statuses
                        st.session_state.incident_status_widget_selection_controlled = [SELECT_ALL_STATUSES_OPTION]
                    elif not unique_statuses: # If there are no statuses at all
                        st.session_state.selected_statuses = []
                        st.session_state.incident_status_widget_selection_controlled = []
                    else: # Default statuses are a specific subset
                        st.session_state.incident_status_widget_selection_controlled = list(default_selected_statuses)

                options_for_status_widget = [SELECT_ALL_STATUSES_OPTION] + unique_statuses
                raw_status_widget_selection = st.multiselect(
                    "Filter by Status",
                    options=options_for_status_widget,
                    default=st.session_state.incident_status_widget_selection_controlled,
                    key="multi_select_incident_status"
                )

                prev_widget_display_state = list(st.session_state.incident_status_widget_selection_controlled)
                current_select_all_option_selected = SELECT_ALL_STATUSES_OPTION in raw_status_widget_selection
                currently_selected_actual_items = [s for s in raw_status_widget_selection if s != SELECT_ALL_STATUSES_OPTION]

                user_clicked_select_all = current_select_all_option_selected and (SELECT_ALL_STATUSES_OPTION not in prev_widget_display_state)
                user_clicked_unselect_all = (not current_select_all_option_selected) and (SELECT_ALL_STATUSES_OPTION in prev_widget_display_state and len(prev_widget_display_state) == 1)

                if user_clicked_select_all:
                    st.session_state.selected_statuses = list(unique_statuses)
                    st.session_state.incident_status_widget_selection_controlled = [SELECT_ALL_STATUSES_OPTION]
                elif user_clicked_unselect_all:
                    st.session_state.selected_statuses = []
                    st.session_state.incident_status_widget_selection_controlled = []
                else:
                    if current_select_all_option_selected:
                        if len(currently_selected_actual_items) < len(unique_statuses):
                            st.session_state.selected_statuses = list(currently_selected_actual_items)
                            st.session_state.incident_status_widget_selection_controlled = list(currently_selected_actual_items)
                        else:
                            st.session_state.selected_statuses = list(unique_statuses)
                            st.session_state.incident_status_widget_selection_controlled = [SELECT_ALL_STATUSES_OPTION]
                    else:
                        st.session_state.selected_statuses = list(currently_selected_actual_items)
                        if unique_statuses and set(currently_selected_actual_items) == set(unique_statuses):
                            st.session_state.incident_status_widget_selection_controlled = [SELECT_ALL_STATUSES_OPTION]
                        else:
                            st.session_state.incident_status_widget_selection_controlled = list(currently_selected_actual_items)

            # Apply filters
            filtered_overview_df = overview_df

            if st.session_state.get('selected_creators') and 'Creator' in filtered_overview_df.columns:
                filtered_overview_df = filtered_overview_df[filtered_overview_df['Creator'].isin(st.session_state.selected_creators)]
            if st.session_state.get('selected_teams') and 'Team' in filtered_overview_df.columns:
                filtered_overview_df = filtered_overview_df[filtered_overview_df['Team'].isin(st.session_state.selected_teams)]
            if st.session_state.get('selected_priorities') and 'Priority' in filtered_overview_df.columns:
                filtered_overview_df = filtered_overview_df[filtered_overview_df['Priority'].isin(st.session_state.selected_priorities)]
            if st.session_state.get('selected_statuses') and 'Status' in filtered_overview_df.columns: # Add status filter
                filtered_overview_df = filtered_overview_df[filtered_overview_df['Status'].isin(st.session_state.selected_statuses)]

            # Calculate team and status totals
            team_status_summary_df = calculate_team_status_summary(filtered_overview_df)

                    # --- Pie Chart for Closed Incidents ---
            st.markdown("---") # Visual separator before the pie chart
            if 'Status' in overview_df.columns: # Ensure 'Status' column exists in the original overview_df
                closed_count = overview_df[overview_df['Status'] == 'Closed'].shape[0]
                total_incidents = overview_df.shape[0]
                other_count = total_incidents - closed_count

                if total_incidents > 0: # Avoid division by zero if no incidents
                    chart_data = pd.DataFrame({
                        'Status Category': ['Closed', 'Open/Other'],
                        'Count': [closed_count, other_count]
                    })
                    fig_status_pie = px.pie(chart_data, names='Status Category', values='Count', title='Percentage of Closed Incidents')
                    st.plotly_chart(fig_status_pie, use_container_width=True)
                else:
                    st.info("No incident data available to display the status pie chart.")
            else:
                st.warning("Cannot display Percentage of Closed Incidents: 'Status' column missing from source data.")
        # --- Team Assignment Distribution ---
        st.markdown("---") # Visual separator
        st.subheader("Team Assignment Distribution")
        if not filtered_overview_df.empty:
            if 'Team' in filtered_overview_df.columns:
                team_distribution_data = filtered_overview_df['Team'].value_counts()
                
                if not team_distribution_data.empty:
                    fig_team_dist = px.pie(
                        team_distribution_data,
                        names=team_distribution_data.index,
                        values=team_distribution_data.values,
                        title="Distribution of Incidents by Team"
                    )
                    fig_team_dist.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_team_dist, use_container_width=True)
                else:
                    st.info("No team assignment data to display based on current filters.")
            else:
                st.warning("Cannot display Team Assignment Distribution: 'Team' column not found in the data.")
        else:
            st.info("No data available to display for Team Assignment Distribution based on current filters.")
        # --- Incidents by Team and Status Table ---
        st.markdown("---") # Visual separator
        st.subheader("Incidents by Team and Status")
        # Check if 'Team' or 'Status' column was missing when team_status_summary_df was created.
        # This check is based on the columns available in filtered_overview_df, which was used to create team_status_summary_df.
        if 'Team' not in filtered_overview_df.columns or 'Status' not in filtered_overview_df.columns:
            st.warning("The 'Team' or 'Status' column is missing in the uploaded incident data, so the 'Incidents by Team and Status' table cannot be generated.")
        elif not team_status_summary_df.empty:
            st.dataframe(team_status_summary_df, use_container_width=True, hide_index=True)
        else:
            # This case means columns existed, but the dataframe is empty (e.g., due to filters or no matching data)
            st.info("No incident data to display in the 'Incidents by Team and Status' table based on current filters or data availability.")
         
        # --- New Filtered Incident Details Table ---
        st.markdown("---") # Separator before the new table
        st.subheader("Filtered Incident Details")

        if not filtered_overview_df.empty:
            # Define default columns for the table
            default_table_columns = ["Incident", "Creator", "Team", "Priority", "Status"]

            # Filter default columns to only include those present in the DataFrame
            available_default_columns = [col for col in default_table_columns if col in filtered_overview_df.columns]

            # Add column selector multiselect
            if not filtered_overview_df.empty:
                all_available_columns = filtered_overview_df.columns.tolist()
                SELECT_ALL_COLS_INCIDENT_OPTION = "[Select All Columns]"

                # available_default_columns is defined above this block in the original code
                if 'incident_tab_column_widget_selection_controlled' not in st.session_state:
                    st.session_state.selected_table_columns = list(available_default_columns) # Actual columns for table

                    if not available_default_columns and all_available_columns: # If no defaults specified but columns exist, select all
                        st.session_state.selected_table_columns = list(all_available_columns)
                        st.session_state.incident_tab_column_widget_selection_controlled = [SELECT_ALL_COLS_INCIDENT_OPTION]
                    elif all_available_columns and set(available_default_columns) == set(all_available_columns): # If defaults are all columns
                        st.session_state.incident_tab_column_widget_selection_controlled = [SELECT_ALL_COLS_INCIDENT_OPTION]
                    elif not all_available_columns: # No columns available at all
                        st.session_state.selected_table_columns = []
                        st.session_state.incident_tab_column_widget_selection_controlled = []
                    else: # Defaults are a specific subset
                        st.session_state.incident_tab_column_widget_selection_controlled = list(available_default_columns)

                options_for_incident_cols_widget = [SELECT_ALL_COLS_INCIDENT_OPTION] + all_available_columns
                raw_incident_cols_widget_selection = st.multiselect(
                    "Select columns for table:",
                    options=options_for_incident_cols_widget,
                    default=st.session_state.incident_tab_column_widget_selection_controlled,
                    key="multi_select_incident_overview_columns"
                )

                prev_widget_display_state = list(st.session_state.incident_tab_column_widget_selection_controlled)
                current_select_all_option_selected = SELECT_ALL_COLS_INCIDENT_OPTION in raw_incident_cols_widget_selection
                currently_selected_actual_items = [c for c in raw_incident_cols_widget_selection if c != SELECT_ALL_COLS_INCIDENT_OPTION]

                user_clicked_select_all = current_select_all_option_selected and (SELECT_ALL_COLS_INCIDENT_OPTION not in prev_widget_display_state)
                user_clicked_unselect_all = (not current_select_all_option_selected) and (SELECT_ALL_COLS_INCIDENT_OPTION in prev_widget_display_state and len(prev_widget_display_state) == 1)

                if user_clicked_select_all:
                    st.session_state.selected_table_columns = list(all_available_columns)
                    st.session_state.incident_tab_column_widget_selection_controlled = [SELECT_ALL_COLS_INCIDENT_OPTION]
                elif user_clicked_unselect_all:
                    st.session_state.selected_table_columns = []
                    st.session_state.incident_tab_column_widget_selection_controlled = []
                else:
                    if current_select_all_option_selected:
                        if len(currently_selected_actual_items) < len(all_available_columns):
                            st.session_state.selected_table_columns = list(currently_selected_actual_items)
                            st.session_state.incident_tab_column_widget_selection_controlled = list(currently_selected_actual_items)
                        else:
                            st.session_state.selected_table_columns = list(all_available_columns)
                            st.session_state.incident_tab_column_widget_selection_controlled = [SELECT_ALL_COLS_INCIDENT_OPTION]
                    else:
                        st.session_state.selected_table_columns = list(currently_selected_actual_items)
                        if all_available_columns and set(currently_selected_actual_items) == set(all_available_columns):
                            st.session_state.incident_tab_column_widget_selection_controlled = [SELECT_ALL_COLS_INCIDENT_OPTION]
                        else:
                            st.session_state.incident_tab_column_widget_selection_controlled = list(currently_selected_actual_items)

                current_cols_to_display_incident_tab = st.session_state.get('selected_table_columns', [])

                if not current_cols_to_display_incident_tab: # Check based on the processed selection
                    st.info("Please select at least one column to display the table.")
                else:
                    # Check if essential columns (even if not selected) are missing from the original data source for context
                    essential_source_cols = ["Incident", "Status"] # Example essentials for the table's purpose
                    missing_essential_source_cols = [col for col in essential_source_cols if col not in filtered_overview_df.columns] # Check against filtered_overview_df
                    if missing_essential_source_cols:
                        st.caption(f"Warning: Source data is missing essential columns for full detail: {', '.join(missing_essential_source_cols)}.")

                    st.write(f"Displaying {len(filtered_overview_df)} records in table with selected columns.") # This len is of the df, not cols
                    st.dataframe(
                        filtered_overview_df[current_cols_to_display_incident_tab],
                        use_container_width=True,
                        hide_index=True
                    )
            else: # This else corresponds to 'if not filtered_overview_df.empty:' for the multiselect definition
                # If filtered_overview_df is empty, no column selector or table is shown.
                # This part of the original code is outside the SEARCH block, so it's retained.
                # The st.info message about "No data to display in the 'Filtered Incident Details' table"
                # will be shown from the outer 'else' block.
                pass # Explicitly pass if no specific action for empty df here regarding column selector

        else:
            st.info("No data to display in the 'Filtered Incident Details' table based on current filters.")
   
        # --- High-Priority Incidents Table (remains, now affected by Status filter too) ---
        st.markdown("---")
        st.subheader("High-Priority Incidents (P1 & P2)")
        if not filtered_overview_df.empty:
            # Include "Status" in this table as well, if available
            high_priority_table_cols = ["Incident", "Creator", "Team", "Priority"]
            if 'Status' in filtered_overview_df.columns:
                high_priority_table_cols.append("Status")

            # Check if all *intended* columns for this table are present
            missing_cols_for_high_priority_table = [col for col in ["Incident", "Creator", "Team", "Priority"] if col not in filtered_overview_df.columns]

            if not missing_cols_for_high_priority_table:
                high_priority_values = ["1", "2"]
                
                high_priority_incidents_df = filtered_overview_df[
                    filtered_overview_df['Priority'].astype(str).isin(high_priority_values)
                ]
                
                if not high_priority_incidents_df.empty:
                    st.dataframe(
                        high_priority_incidents_df[[col for col in high_priority_table_cols if col in high_priority_incidents_df.columns]], # Display only available columns
                        use_container_width=True,
                        hide_index=True 
                    )
                else:
                    st.info("No high-priority incidents (P1 or P2) found based on current filters.")
            else:
                st.warning(f"Cannot display High-Priority Incidents table: Missing essential columns: {', '.join(missing_cols_for_high_priority_table)}.")
        else:
            st.info("No data available to display for High-Priority Incidents based on current filters.")

st.markdown("---")
st.markdown(
    """<div style="text-align:center; color:#888; font-size:0.8em;">
    Intellipen SmartQ V3.5 | Developed by Ali Babiker | © June 2025
    </div>""",
    unsafe_allow_html=True
)
