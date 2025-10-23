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
from utils import calculate_team_status_summary, calculate_srs_created_per_week, _get_week_display_str, extract_approver_name # Added _get_week_display_str

# Set page configuration
st.set_page_config(
    page_title="Intellipen SmartQ Test",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import the new function from utils
from utils import calculate_incidents_breached_per_week

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

    /* Dark Mode Styles */
    @media (prefers-color-scheme: dark) {
        .main {
            background-color: #0e1117; /* Streamlit's default dark background */
            color: #fafafa; /* Light text for dark background */
        }
        .stApp {
            color: #fafafa;
        }
        .stDataFrame, .stTable {
            background-color: #262730; /* Darker background for tables */
            color: #fafafa;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.4), 0 1px 2px rgba(0,0,0,0.5);
        }
        .card {
            background-color: #262730; /* Darker card background */
            color: #fafafa;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.4), 0 1px 2px rgba(0,0,0,0.5);
            margin-bottom: 20px;
        }
        .metric-value {
            color: #fafafa;
        }
        .metric-label {
            color: #aaa; /* Lighter grey for labels */
        }
        h1, h2, h3 {
            color: #5c9dff; /* A lighter blue for headings */
        }
        /* Adjust badge colors for dark mode */
        .badge-pending {
            background-color: #533f00; /* Darker yellow */
            color: #ffecb3;
        }
        .badge-complete {
            background-color: #1b5e20; /* Darker green */
            color: #c8e6c9;
        }
        .badge-in-progress {
            background-color: #0d47a1; /* Darker blue */
            color: #bbdefb;
        }
        .badge-cancelled {
            background-color: #b71c1c; /* Darker red */
            color: #ffcdd2;
        }
        .badge-breach {
            background-color: #d32f2f; /* Slightly different red for breach */
            color: white;
        }
        .action-button {
            background-color: #5c9dff; /* Lighter blue for buttons */
            color: #0e1117; /* Dark text for light buttons */
        }
        .action-button:hover {
            background-color: #4a80cc; /* Darker shade of the button blue for hover */
        }

        /* Ensure selectbox and multiselect dropdowns are styled for dark mode */
        div[data-baseweb="select"] > div, div[data-baseweb="multiselect"] > div {
            background-color: #262730 !important;
            color: #fafafa !important;
            border: 1px solid #555 !important;
        }
        div[data-baseweb="select"] input, div[data-baseweb="multiselect"] input {
            color: #fafafa !important;
        }
        /* Dropdown menu items */
        div[data-baseweb="menu"] li {
            background-color: #262730 !important;
            color: #fafafa !important;
        }
        div[data-baseweb="menu"] li:hover {
            background-color: #383943 !important;
        }

        /* Ensure text inputs are styled for dark mode */
        .stTextInput input, .stTextArea textarea, .stDateInput input {
            background-color: #262730 !important;
            color: #fafafa !important;
            border: 1px solid #555 !important;
        }

        /* Style for option menu in dark mode */
        nav[role="tablist"] > a {
            color: #bbb !important; /* Lighter text for non-selected tabs */
        }
        nav[role="tablist"] > a.nav-link-selected {
            background-color: #5c9dff !important; /* Selected tab background */
            color: #0e1117 !important; /* Dark text on selected tab */
        }
        nav[role="tablist"] > a:hover {
            background-color: #383943 !important; /* Hover for non-selected tabs */
            color: #fafafa !important;
        }

        /* Ensure Streamlit buttons are styled */
        .stButton>button {
            background-color: #5c9dff !important;
            color: #0e1117 !important;
            border: 1px solid #5c9dff !important;
        }
        .stButton>button:hover {
            background-color: #4a80cc !important;
            border: 1px solid #4a80cc !important;
            color: #0e1117 !important;
        }
        .stDownloadButton>button {
            background-color: #5c9dff !important;
            color: #0e1117 !important;
            border: 1px solid #5c9dff !important;
        }
        .stDownloadButton>button:hover {
            background-color: #4a80cc !important;
            border: 1px solid #4a80cc !important;
            color: #0e1117 !important;
        }
        /* Sidebar styling */
        .css-1d391kg { /* This class might be Streamlit version specific, targets sidebar */
            background-color: #1a1c22 !important;
        }
        .css-1d391kg .stMarkdown, .css-1d391kg .stSubheader, .css-1d391kg .stTitle {
             color: #fafafa !important;
        }
         /* Adjusting table header in dark mode */
        div[data-testid="stDataFrame"] th {
            background-color: #383943 !important; /* Darker header for tables */
            color: #fafafa !important;
        }
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
    # Drop duplicates based on 'Case Id', keeping the last occurrence
    if 'Case Id' in df.columns:
        initial_rows = len(df)
        # Use inplace=False to return a new DataFrame, which is safer.
        df = df.drop_duplicates(subset=['Case Id'], keep='last')
        rows_dropped = initial_rows - len(df)
        if rows_dropped > 0:
            print(f"--- INFO: Dropped {rows_dropped} duplicate cases based on 'Case Id'. ---")

    # Ensure date columns are in datetime format
    date_columns = ['Case Start Date', 'Last Note Date']
    for col in date_columns:
        if col in df.columns:
            # Use .loc to avoid SettingWithCopyWarning
            df.loc[:, col] = pd.to_datetime(df[col], format="%d/%m/%Y", errors='coerce')
    
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
        ticket_keyword = match.group(1).lower()

        sr_keywords = ['sr', 'مرجعي', 'اس ار']

        ticket_type = "SR" if ticket_keyword in sr_keywords else "Incident"
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
    st.title("📊 Intellipen SmartQ Test")
    st.markdown("---")

    st.subheader("📁 Data Import")
    uploaded_file = st.file_uploader("Upload Main Excel File (.xlsx)", type=["xlsx"])
    sr_status_file = st.file_uploader("Upload SR Status Excel (optional)", type=["xlsx"])
    incident_status_file = st.file_uploader("Upload Incident Report Excel (optional)", type=["xlsx"])
    
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
                # Attempt to parse 'Breach Date' right after loading incident_overview_df
                if 'Breach Date' in overview_df.columns:
                    col_name = 'Breach Date'
                    original_series = overview_df[col_name].copy().astype(str) # Work with string representations
                    overview_df[col_name] = pd.NaT # Initialize with NaT

                    # Step 1: Try specific known formats (day-first)
                    formats_to_try = ['%d/%m/%Y %H:%M:%S', '%d/%m/%y %H:%M', '%d/%m/%Y']
                    for fmt in formats_to_try:
                        mask = overview_df[col_name].isnull() & original_series.notnull()
                        if not mask.any(): break
                        parsed_subset = pd.to_datetime(original_series[mask], format=fmt, errors='coerce')
                        overview_df.loc[mask, col_name] = overview_df.loc[mask, col_name].fillna(parsed_subset)

                    # Step 2: Try standard parsing (handles ISO, etc.) for remaining nulls
                    mask = overview_df[col_name].isnull() & original_series.notnull()
                    if mask.any():
                        iso_parsed = pd.to_datetime(original_series[mask], errors='coerce')
                        overview_df.loc[mask, col_name] = overview_df.loc[mask, col_name].fillna(iso_parsed)

                    # Step 3: Try general dayfirst=True parsing for remaining nulls
                    mask = overview_df[col_name].isnull() & original_series.notnull()
                    if mask.any():
                        dayfirst_gen_parsed = pd.to_datetime(original_series[mask], errors='coerce', dayfirst=True)
                        overview_df.loc[mask, col_name] = overview_df.loc[mask, col_name].fillna(dayfirst_gen_parsed)

                    # Optionally, log how many failed to parse if necessary, or notify user.
                    # For now, we proceed with successfully parsed dates.

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
        default_users_hardcoded = ['ali.babiker', 'anas.hasan', 'ahmed.mostafa','GPSSA_H.Salah','alharith.alfki']
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
    st.title("📊 Intellipen SmartQ Test")
    st.markdown("""
    ### Welcome to the Intellipen SmartQ Test!
    
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
    
    # Apply date filter if date range is selected and column exists
    if 'date_range' in locals() and 'Case Start Date' in df_filtered.columns:
        # Ensure date_range is a tuple of two dates
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            # Additional check to ensure dates are not NaT or None
            if pd.notna(start_date) and pd.notna(end_date):
                # Ensure the 'Case Start Date' column is in datetime format before filtering
                if pd.api.types.is_datetime64_any_dtype(df_filtered['Case Start Date']):
                    df_filtered = df_filtered[
                        (df_filtered['Case Start Date'].dt.date >= start_date) &
                        (df_filtered['Case Start Date'].dt.date <= end_date)
                    ]
    
    # Prepare tab interface
    selected = option_menu(
        menu_title=None,
        options=["Analysis", "SLA Breach", "Today's SR/Incidents", "Incident Overview", "SR Overview"],
        icons=["kanban", "exclamation-triangle", "calendar-date", "clipboard-data", "bar-chart-line"],
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
        df_enriched['Pending With'] = None
        
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

                # Proactively rename columns from the SR file to avoid suffix ambiguity
                sr_cols_to_rename = {col: f"{col}_sr" for col in sr_df_copy.columns if col != 'Service Request'}
                sr_df_copy.rename(columns=sr_cols_to_rename, inplace=True)

                # Merge all columns from sr_df
                df_enriched = df_enriched.merge(
                    sr_df_copy,
                    how='left',
                    left_on='Ticket Number',
                    right_on='Service Request'
                    # No suffix needed now as columns are pre-renamed
                )

                sr_mask = df_enriched['Type'] == 'SR'

                # Populate unified columns from the suffixed SR columns
                if 'Status_sr' in df_enriched.columns:
                    df_enriched.loc[sr_mask, 'Status'] = df_enriched.loc[sr_mask, 'Status_sr']
                if 'LastModDateTime_sr' in df_enriched.columns:
                    df_enriched.loc[sr_mask, 'Last Update'] = df_enriched.loc[sr_mask, 'LastModDateTime_sr']

                if 'Breach Passed_sr' in df_enriched.columns:
                    def map_str_to_bool_sr(value):
                        if pd.isna(value): return None
                        val_lower = str(value).lower()
                        if val_lower in ['yes', 'true', '1', 'passed'] : return True
                        if val_lower in ['no', 'false', '0', 'failed']: return False
                        return None

                    mapped_values = df_enriched.loc[sr_mask, 'Breach Passed_sr'].apply(map_str_to_bool_sr)
                    df_enriched.loc[sr_mask, 'Breach Passed'] = mapped_values

                if 'Approval Pending with_sr' in df_enriched.columns:
                    df_enriched.loc[sr_mask, 'Pending With'] = df_enriched.loc[sr_mask, 'Approval Pending with_sr'].apply(extract_approver_name)

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
            col_name_en = 'Breach Date'
            original_series_en = df_enriched[col_name_en].copy().astype(str)
            df_enriched[col_name_en] = pd.NaT

            # Step 1: Try specific known formats (day-first)
            formats_to_try = ['%d/%m/%Y %H:%M:%S', '%d/%m/%y %H:%M', '%d/%m/%Y']
            for fmt in formats_to_try:
                mask_en = df_enriched[col_name_en].isnull() & original_series_en.notnull()
                if not mask_en.any(): break
                parsed_subset_en = pd.to_datetime(original_series_en[mask_en], format=fmt, errors='coerce')
                df_enriched.loc[mask_en, col_name_en] = df_enriched.loc[mask_en, col_name_en].fillna(parsed_subset_en)

            # Step 2: Try standard parsing (handles ISO, etc.) for remaining nulls
            mask_en = df_enriched[col_name_en].isnull() & original_series_en.notnull()
            if mask_en.any():
                iso_parsed_en = pd.to_datetime(original_series_en[mask_en], errors='coerce')
                df_enriched.loc[mask_en, col_name_en] = df_enriched.loc[mask_en, col_name_en].fillna(iso_parsed_en)

            # Step 3: Try general dayfirst=True parsing for remaining nulls
            mask_en = df_enriched[col_name_en].isnull() & original_series_en.notnull()
            if mask_en.any():
                dayfirst_gen_parsed_en = pd.to_datetime(original_series_en[mask_en], errors='coerce', dayfirst=True)
                df_enriched.loc[mask_en, col_name_en] = df_enriched.loc[mask_en, col_name_en].fillna(dayfirst_gen_parsed_en)

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
                    status_all_counts = df_srs_status_valid['Status'].value_counts().rename_axis('Status').reset_index(name='Cases Count')
                    
                    # Unique SR tickets
                    ticket_unique = df_srs_status_valid.dropna(subset=['Ticket Number'])[['Ticket Number', 'Status']].drop_duplicates()
                    ticket_unique_counts = ticket_unique['Status'].value_counts().rename_axis('Status').reset_index(name='SR Count')
                    
                    # Merge both summaries
                    merged_status = pd.merge(status_all_counts, ticket_unique_counts, on='Status', how='outer').fillna(0)
                    merged_status[['Cases Count', 'SR Count']] = merged_status[['Cases Count', 'SR Count']].astype(int)
                    
                    # New logic for breakdown
                    new_rows = []
                    for _, row in merged_status.iterrows():
                        new_rows.append(row)
                        # Use case-insensitive and whitespace-agnostic comparison
                        if str(row['Status']).strip().lower() == 'waiting for approval':
                            # Get the breakdown for this status
                            srs_waiting = df_srs_status_valid[df_srs_status_valid['Status'].apply(lambda x: str(x).strip().lower() == 'waiting for approval')]

                            if not srs_waiting.empty and 'Pending With' in srs_waiting.columns:
                                # Breakdown for cases
                                case_breakdown = srs_waiting['Pending With'].value_counts().reset_index()
                                case_breakdown.columns = ['Pending With', 'Cases Count']

                                # Breakdown for unique SRs
                                sr_breakdown = srs_waiting.drop_duplicates(subset=['Ticket Number'])['Pending With'].value_counts().reset_index()
                                sr_breakdown.columns = ['Pending With', 'SR Count']

                                # Merge breakdowns
                                final_breakdown = pd.merge(case_breakdown, sr_breakdown, on='Pending With', how='outer').fillna(0)

                                for _, breakdown_row in final_breakdown.iterrows():
                                    if pd.notna(breakdown_row['Pending With']):
                                        new_row = {
                                            'Status': f"    \u21b3 {breakdown_row['Pending With']}", # Indent with spaces and an arrow
                                            'Cases Count': int(breakdown_row['Cases Count']),
                                            'SR Count': int(breakdown_row['SR Count'])
                                        }
                                        new_rows.append(pd.Series(new_row))

                    # Create the new summary dataframe
                    if new_rows:
                        status_summary_df_with_breakdown = pd.DataFrame(new_rows)
                    else:
                        status_summary_df_with_breakdown = merged_status.copy()

                    # Total row
                    total_row = {
                        'Status': 'Total',
                        'Cases Count': merged_status['Cases Count'].sum(),
                        'SR Count': merged_status['SR Count'].sum()
                    }
                    
                    status_summary_df = pd.concat([status_summary_df_with_breakdown, pd.DataFrame([total_row])], ignore_index=True)
                    
                    # Display with original styling
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
                    incident_ticket_unique_counts = incident_ticket_unique['Status'].value_counts().rename_axis('Status').reset_index(name='Incident Count')
                    
                    # Merge both summaries for incidents
                    merged_incident_status = pd.merge(incident_status_all_counts, incident_ticket_unique_counts, on='Status', how='outer').fillna(0)
                    merged_incident_status[['Cases Count', 'Incident Count']] = merged_incident_status[['Cases Count', 'Incident Count']].astype(int)
                    
                    # Total row for incidents
                    incident_total_row = {
                        'Status': 'Total',
                        'Cases Count': merged_incident_status['Cases Count'].sum(),
                        'Incident Count': merged_incident_status['Incident Count'].sum()
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

        # Define PAP category specifics early to ensure they are in scope
        pap_category_column = 'Category'
        pap_category_value = 'Pension Application Platform (PAP)'

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

                # Define the desired default teams
                default_teams_to_select = ["GPSSA App Team L1", "GPSSA App Team L3", "GPSSA PS Team L3"]

                if 'incident_team_widget_selection_controlled' not in st.session_state:
                    # Filter desired default teams to only those present in unique_teams
                    actual_default_teams = [team for team in default_teams_to_select if team in unique_teams]

                    if actual_default_teams:
                        st.session_state.selected_teams = list(actual_default_teams)
                        # Check if all unique_teams are selected by the new default
                        if unique_teams and set(actual_default_teams) == set(unique_teams):
                            st.session_state.incident_team_widget_selection_controlled = [SELECT_ALL_TEAMS_OPTION]
                        else:
                            st.session_state.incident_team_widget_selection_controlled = list(actual_default_teams)
                    elif unique_teams: # If desired defaults are not present, but other teams are, select all (original fallback)
                        st.session_state.selected_teams = list(unique_teams)
                        st.session_state.incident_team_widget_selection_controlled = [SELECT_ALL_TEAMS_OPTION]
                    else: # No teams in data, or desired defaults not present and no other teams
                        st.session_state.selected_teams = []
                        st.session_state.incident_team_widget_selection_controlled = []

                options_for_team_widget = [SELECT_ALL_TEAMS_OPTION] + unique_teams
                # The default for the multiselect widget is now correctly initialized in session state
                raw_team_widget_selection = st.multiselect(
                    "Filter by Team",
                    options=options_for_team_widget,
                    default=st.session_state.incident_team_widget_selection_controlled, # This uses the initialized value
                    key="multi_select_incident_team"
                )

                prev_widget_display_state = list(st.session_state.incident_team_widget_selection_controlled) # Used for "Select All" logic
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

        # --- Charts Display: Percentage of Closed Incidents and Team Assignment Distribution ---
        # Use columns to display charts side-by-side
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("Percentage of Closed Incidents")
            if 'Status' in overview_df.columns: # Ensure 'Status' column exists in the original overview_df for this chart
                closed_count = overview_df[overview_df['Status'] == 'Closed'].shape[0]
                total_incidents = overview_df.shape[0]
                other_count = total_incidents - closed_count

                if total_incidents > 0: # Avoid division by zero if no incidents
                    chart_data_status_pie = pd.DataFrame({ # Use a unique variable name for this chart's data
                        'Status Category': ['Closed', 'Open/Other'],
                        'Count': [closed_count, other_count]
                    })
                    # Pass a unique key to this plotly_chart instance
                    fig_status_pie = px.pie(chart_data_status_pie, names='Status Category', values='Count', title='Percentage of Closed Incidents')
                    st.plotly_chart(fig_status_pie, use_container_width=True, key="status_pie_chart")
                else:
                    st.info("No incident data available to display the status pie chart.")
            else:
                st.warning("Cannot display Percentage of Closed Incidents: 'Status' column missing from source data (for status pie chart).")

        with chart_col2:
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

        st.markdown("---") # Visual separator

        # --- Incidents Breached Per Week Graph ---
        st.subheader("Incidents Breached Per Week")
        if 'incident_overview_df' in st.session_state and st.session_state.incident_overview_df is not None and not st.session_state.incident_overview_df.empty:
            # Make a copy to avoid modifying the session state DataFrame directly during processing
            inc_breach_df_source = st.session_state.incident_overview_df.copy()

            if 'Breach Date' in inc_breach_df_source.columns:
                # Ensure 'Breach Date' is present before calling the utility function
                # The utility function itself handles parsing and errors for 'Breach Date'
                weekly_breached_incidents_df = calculate_incidents_breached_per_week(inc_breach_df_source, breach_date_col='Breach Date')

                if not weekly_breached_incidents_df.empty:
                    fig_incidents_breached = px.bar(
                        weekly_breached_incidents_df,
                        x='WeekDisplay',
                        y='Count',
                        title="Incidents Breached Per Week",
                        labels={'Count': 'Number of Incidents Breached', 'WeekDisplay': 'Week Period'},
                        text='Count'
                    )
                    fig_incidents_breached.update_traces(texttemplate='%{text}', textposition='outside')
                    fig_incidents_breached.update_layout(xaxis_title='Week Period', yaxis_title="Number of Incidents Breached")
                    st.plotly_chart(fig_incidents_breached, use_container_width=True)
                else:
                    st.info("No data available to display the 'Incidents Breached Per Week' chart. This may be due to missing or invalid 'Breach Date' values in the uploaded incident data.")
            else:
                st.warning("The 'Breach Date' column is missing from the uploaded incident data. Cannot generate 'Incidents Breached Per Week' chart.")
        else:
            st.info("Incident data not loaded or empty. Please upload incident data to see the breached incidents chart.")

        st.markdown("---") # Visual separator

        # --- Detailed Breached Incidents Table with Column Selection ---
        st.subheader("Detailed Breached Incidents")
        if 'incident_overview_df' in st.session_state and \
           st.session_state.incident_overview_df is not None and \
           not st.session_state.incident_overview_df.empty:

            detailed_breach_source_df = st.session_state.incident_overview_df.copy()

            if 'Breach Date' in detailed_breach_source_df.columns:
                col_name_detail = 'Breach Date'
                parsed_col_name_detail = 'Breach Date Parsed'
                original_series_detail = detailed_breach_source_df[col_name_detail].copy().astype(str)
                detailed_breach_source_df[parsed_col_name_detail] = pd.NaT

                # Step 1: Try specific known formats (day-first)
                formats_to_try = ['%d/%m/%Y %H:%M:%S', '%d/%m/%y %H:%M', '%d/%m/%Y']
                for fmt in formats_to_try:
                    mask_detail = detailed_breach_source_df[parsed_col_name_detail].isnull() & original_series_detail.notnull()
                    if not mask_detail.any(): break
                    parsed_subset_detail = pd.to_datetime(original_series_detail[mask_detail], format=fmt, errors='coerce')
                    detailed_breach_source_df.loc[mask_detail, parsed_col_name_detail] = detailed_breach_source_df.loc[mask_detail, parsed_col_name_detail].fillna(parsed_subset_detail)

                # Step 2: Try standard parsing (handles ISO, etc.) for remaining nulls
                mask_detail = detailed_breach_source_df[parsed_col_name_detail].isnull() & original_series_detail.notnull()
                if mask_detail.any():
                    iso_parsed_detail = pd.to_datetime(original_series_detail[mask_detail], errors='coerce')
                    detailed_breach_source_df.loc[mask_detail, parsed_col_name_detail] = detailed_breach_source_df.loc[mask_detail, parsed_col_name_detail].fillna(iso_parsed_detail)

                # Step 3: Try general dayfirst=True parsing for remaining nulls
                mask_detail = detailed_breach_source_df[parsed_col_name_detail].isnull() & original_series_detail.notnull()
                if mask_detail.any():
                    dayfirst_gen_parsed_detail = pd.to_datetime(original_series_detail[mask_detail], errors='coerce', dayfirst=True)
                    detailed_breach_source_df.loc[mask_detail, parsed_col_name_detail] = detailed_breach_source_df.loc[mask_detail, parsed_col_name_detail].fillna(dayfirst_gen_parsed_detail)

                # Filter for incidents that have a valid (parsed) breach date
                all_breached_incidents_df = detailed_breach_source_df.dropna(subset=['Breach Date Parsed'])

                # We can drop the temporary parsed column if we show the original 'Breach Date'
                # However, it might be useful to keep the parsed one for sorting or display consistency.
                # For now, let's assume we want to display columns from the original df structure.
                # So, we use 'all_breached_incidents_df' to identify rows, but select columns from 'detailed_breach_source_df' (before dropping 'Breach Date Parsed') or even better, from original `overview_df` by index.

                # Let's use the indices from `all_breached_incidents_df` to filter the original `st.session_state.incident_overview_df`
                # to ensure we are showing original data and all its columns.

                displayable_breached_incidents_df = st.session_state.incident_overview_df.loc[all_breached_incidents_df.index].copy() # Use .copy() to avoid SettingWithCopyWarning

                # Ensure 'Breach Date' is datetime for further operations (like deriving Year-Week or day filtering)
                # This should already be the case due to earlier parsing, but explicitly ensuring it here if it's re-read or copied.
                if 'Breach Date' in displayable_breached_incidents_df.columns:
                    displayable_breached_incidents_df['Breach Date'] = pd.to_datetime(displayable_breached_incidents_df['Breach Date'], errors='coerce')
                    # Add 'Year-Week' column for filtering, if 'Breach Date' is valid datetime
                    if pd.api.types.is_datetime64_any_dtype(displayable_breached_incidents_df['Breach Date']):
                        displayable_breached_incidents_df['Year-Week'] = displayable_breached_incidents_df['Breach Date'].dt.strftime('%G-W%V')
                    else:
                        # Fallback if 'Breach Date' is not datetime (should not happen if parsing worked)
                        displayable_breached_incidents_df['Year-Week'] = None
                else:
                     displayable_breached_incidents_df['Year-Week'] = None # Ensure column exists even if 'Breach Date' is missing


                if not displayable_breached_incidents_df.empty:
                    st.markdown(f"Found **{len(displayable_breached_incidents_df)}** incidents with a valid breach date (before applying week/day filters).")

                    # --- Week and Day Filters for Detailed Table (Category filter removed) ---
                    filter_col1, filter_col2 = st.columns(2) # Reverted to 2 columns
                    selected_week_displays_breach = []
                    selected_day_breach = None
                    # selected_categories_breach variable is no longer needed here

                    with filter_col1:
                        if 'weekly_breached_incidents_df' in locals() and not weekly_breached_incidents_df.empty and 'WeekDisplay' in weekly_breached_incidents_df.columns:
                            week_options_breach = ["All Weeks"] + weekly_breached_incidents_df['WeekDisplay'].unique().tolist()
                            if 'breach_week_filter_selection' not in st.session_state:
                                st.session_state.breach_week_filter_selection = ["All Weeks"]

                            selected_week_displays_breach = st.multiselect(
                                "Filter by Week Period (Breach Date):",
                                options=week_options_breach,
                                default=st.session_state.breach_week_filter_selection,
                                key="multiselect_breach_week_period"
                            )
                            st.session_state.breach_week_filter_selection = selected_week_displays_breach
                        else:
                            st.caption("Week filter not available (no weekly breach data).")

                    with filter_col2:
                        if 'Breach Date' in displayable_breached_incidents_df.columns and not displayable_breached_incidents_df['Breach Date'].dropna().empty:
                            min_breach_date = displayable_breached_incidents_df['Breach Date'].dropna().min().date()
                            max_breach_date = displayable_breached_incidents_df['Breach Date'].dropna().max().date()
                            selected_day_breach = st.date_input(
                                "Filter by Specific Day (Breach Date):",
                                value=None,
                                min_value=min_breach_date,
                                max_value=max_breach_date,
                                key="date_input_breach_specific_day"
                            )
                        else:
                            st.caption("Day filter not available (no valid breach dates).")

                    # Prepare for filtering - this df will be further filtered
                    filtered_detailed_breached_incidents_df = displayable_breached_incidents_df.copy()

                    # Apply Day Filter (takes precedence)
                    if selected_day_breach:
                        if 'Breach Date' in filtered_detailed_breached_incidents_df.columns and \
                           pd.api.types.is_datetime64_any_dtype(filtered_detailed_breached_incidents_df['Breach Date']):
                            filtered_detailed_breached_incidents_df = filtered_detailed_breached_incidents_df[
                                filtered_detailed_breached_incidents_df['Breach Date'].dt.date == selected_day_breach
                            ]
                            st.session_state.breach_week_filter_selection = ["All Weeks"]

                    # Apply Week Filter (if no day filter is active and a specific week is chosen)
                    elif selected_week_displays_breach and "All Weeks" not in selected_week_displays_breach:
                        if 'weekly_breached_incidents_df' in locals() and not weekly_breached_incidents_df.empty and \
                           'Year-Week' in weekly_breached_incidents_df.columns and \
                           'Year-Week' in filtered_detailed_breached_incidents_df.columns:

                            week_map_breach = pd.Series(
                                weekly_breached_incidents_df['Year-Week'].values,
                                index=weekly_breached_incidents_df['WeekDisplay']
                            ).to_dict()

                            selected_year_weeks_breach = [
                                week_map_breach[wd] for wd in selected_week_displays_breach if wd in week_map_breach
                            ]

                            if selected_year_weeks_breach:
                                filtered_detailed_breached_incidents_df = filtered_detailed_breached_incidents_df[
                                    filtered_detailed_breached_incidents_df['Year-Week'].isin(selected_year_weeks_breach)
                                ]
                    # Category filter logic removed here

                    # Update the count message after all filters
                    st.markdown(f"Displaying **{len(filtered_detailed_breached_incidents_df)}** '{pap_category_value}' breached incidents based on current filters.")


                    all_breached_incident_columns = filtered_detailed_breached_incidents_df.columns.tolist() # This will include 'Year-Week' if present
                    SELECT_ALL_COLS_BREACH_DETAIL_OPTION = "[Select All Columns for Breached Incidents]"

                    # Add 'Year-Week' to default columns if it exists in the source
                    default_breach_detail_cols = ["Incident", "Creator", "Team", "Priority", "Status", "Breach Date"]
                    if 'Year-Week' in all_breached_incident_columns:
                        default_breach_detail_cols.append('Year-Week')

                    actual_default_breach_detail_cols = [col for col in default_breach_detail_cols if col in all_breached_incident_columns]

                    if 'breached_incident_detail_cols_controlled' not in st.session_state:
                        st.session_state.selected_breach_detail_display_cols = list(actual_default_breach_detail_cols)
                        if not actual_default_breach_detail_cols and all_breached_incident_columns:
                            st.session_state.selected_breach_detail_display_cols = list(all_breached_incident_columns)
                            st.session_state.breached_incident_detail_cols_controlled = [SELECT_ALL_COLS_BREACH_DETAIL_OPTION]
                        elif all_breached_incident_columns and set(actual_default_breach_detail_cols) == set(all_breached_incident_columns):
                            st.session_state.breached_incident_detail_cols_controlled = [SELECT_ALL_COLS_BREACH_DETAIL_OPTION]
                        elif not all_breached_incident_columns:
                            st.session_state.selected_breach_detail_display_cols = []
                            st.session_state.breached_incident_detail_cols_controlled = []
                        else:
                             st.session_state.breached_incident_detail_cols_controlled = list(actual_default_breach_detail_cols)

                    options_for_breach_detail_cols_widget = [SELECT_ALL_COLS_BREACH_DETAIL_OPTION] + all_breached_incident_columns
                    raw_breach_detail_cols_selection = st.multiselect(
                        "Select columns for Detailed Breached Incidents table:",
                        options=options_for_breach_detail_cols_widget,
                        default=st.session_state.breached_incident_detail_cols_controlled,
                        key="multi_select_breached_incident_detail_columns"
                    )

                    # Logic for "Select All" option for breached incident detail columns
                    prev_widget_state_breach_detail = list(st.session_state.breached_incident_detail_cols_controlled)
                    current_select_all_breach_detail = SELECT_ALL_COLS_BREACH_DETAIL_OPTION in raw_breach_detail_cols_selection
                    selected_actual_items_breach_detail = [c for c in raw_breach_detail_cols_selection if c != SELECT_ALL_COLS_BREACH_DETAIL_OPTION]

                    user_clicked_select_all_breach_detail = current_select_all_breach_detail and (SELECT_ALL_COLS_BREACH_DETAIL_OPTION not in prev_widget_state_breach_detail)
                    user_clicked_unselect_all_breach_detail = (not current_select_all_breach_detail) and (SELECT_ALL_COLS_BREACH_DETAIL_OPTION in prev_widget_state_breach_detail and len(prev_widget_state_breach_detail) == 1)

                    if user_clicked_select_all_breach_detail:
                        st.session_state.selected_breach_detail_display_cols = list(all_breached_incident_columns)
                        st.session_state.breached_incident_detail_cols_controlled = [SELECT_ALL_COLS_BREACH_DETAIL_OPTION]
                    elif user_clicked_unselect_all_breach_detail:
                        st.session_state.selected_breach_detail_display_cols = []
                        st.session_state.breached_incident_detail_cols_controlled = []
                    else:
                        if current_select_all_breach_detail:
                            if len(selected_actual_items_breach_detail) < len(all_breached_incident_columns):
                                st.session_state.selected_breach_detail_display_cols = list(selected_actual_items_breach_detail)
                                st.session_state.breached_incident_detail_cols_controlled = list(selected_actual_items_breach_detail)
                            else:
                                st.session_state.selected_breach_detail_display_cols = list(all_breached_incident_columns)
                                st.session_state.breached_incident_detail_cols_controlled = [SELECT_ALL_COLS_BREACH_DETAIL_OPTION]
                        else:
                            st.session_state.selected_breach_detail_display_cols = list(selected_actual_items_breach_detail)
                            if all_breached_incident_columns and set(selected_actual_items_breach_detail) == set(all_breached_incident_columns):
                                st.session_state.breached_incident_detail_cols_controlled = [SELECT_ALL_COLS_BREACH_DETAIL_OPTION]
                            else:
                                st.session_state.breached_incident_detail_cols_controlled = list(selected_actual_items_breach_detail)

                    columns_to_show_breach_detail = st.session_state.get('selected_breach_detail_display_cols', [])

                    if not columns_to_show_breach_detail: # If list is empty (e.g. user unselected all)
                        if actual_default_breach_detail_cols: # Try to show defaults
                            columns_to_show_breach_detail = actual_default_breach_detail_cols
                        elif all_breached_incident_columns: # Else show all available
                            columns_to_show_breach_detail = all_breached_incident_columns
                        # If still empty, the next 'if' handles it

                    if columns_to_show_breach_detail:
                        st.dataframe(filtered_detailed_breached_incidents_df[columns_to_show_breach_detail], hide_index=True, use_container_width=True)
                    else: # This covers cases where filtered_detailed_breached_incidents_df is empty OR no columns ended up in columns_to_show_breach_detail
                        st.info("No data or columns available to display for detailed breached incidents based on current filters and selections.")

                # This 'else' correctly corresponds to 'if not displayable_breached_incidents_df.empty:'
                else:
                    st.info(f"No '{pap_category_value}' incidents with a valid breach date found to display details (after week/day filters or initially).")

            # This 'else' correctly corresponds to 'if 'Breach Date' in pap_incidents_df.columns:'
            # (or rather, the derived working_df_for_detailed_table)
            else:
                st.warning(f"The 'Breach Date' column is missing in the filtered '{pap_category_value}' incident data. Cannot display detailed breached incidents.")

        # This 'elif' and 'else' handle cases where pap_incidents_df itself was empty from the start
        elif not pap_incidents_df.empty and 'Breach Date' not in pap_incidents_df.columns:
             st.warning(f"The 'Breach Date' column is missing from '{pap_category_value}' incidents. Cannot display detailed breaches.")
        else: # pap_incidents_df is empty
            # Check original source only if pap_incidents_df is empty
            if 'incident_overview_df' not in st.session_state or st.session_state.incident_overview_df is None or st.session_state.incident_overview_df.empty:
                st.info("Incident data not loaded or empty. Please upload incident data to see detailed breached incidents.")
            # else: specific messages about why pap_incidents_df is empty (no PAP, or Category col missing) were shown earlier
            # We can add a generic one here if needed for the detailed table context.
            # elif not (st.session_state.incident_overview_df[pap_category_column].astype(str).str.strip().str.lower() == pap_category_value.strip().lower()).any():
            #    st.info(f"No incidents of category '{pap_category_value}' found to display details.")


        st.markdown("---") # Visual separator
        # --- Incidents by Team and Status Table ---
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

    #
    # SR OVERVIEW TAB
    #
    elif selected == "SR Overview":
        st.title("📊 Service Request (SR) Overview")
        # Import the new function
        from utils import calculate_srs_created_and_closed_per_week


        if 'sr_df' not in st.session_state or st.session_state.sr_df is None or st.session_state.sr_df.empty:
            st.warning(
                "The 'SR Status Excel' has not been uploaded or is empty. "
                "Please upload the SR status file via the sidebar to view the SR Overview."
            )
        else:
            sr_overview_df = st.session_state.sr_df.copy()
            st.markdown(f"**Total SRs Loaded:** {len(sr_overview_df)}")

            # Check for required columns for the new chart
            required_cols_for_chart = ['Created On', 'LastModDateTime', 'Status']
            missing_cols = [col for col in required_cols_for_chart if col not in sr_overview_df.columns]

            if missing_cols:
                st.error(f"The SR data must contain the following columns to generate the weekly overview: {', '.join(missing_cols)}.")
            else:
                # Use the new function
                srs_weekly_combined_df = calculate_srs_created_and_closed_per_week(sr_overview_df)

                if srs_weekly_combined_df.empty:
                    st.info("No valid data found to generate the weekly SRs created/closed chart.")
                else:
                    # Filter data for created SRs
                    created_df = srs_weekly_combined_df[srs_weekly_combined_df['Category'] == 'Created'].copy()
                    # Filter data for closed SRs
                    closed_df = srs_weekly_combined_df[srs_weekly_combined_df['Category'] == 'Closed'].copy()
                    
                    chart_x_axis = 'WeekDisplay'

                    if not created_df.empty:
                        st.subheader("Service Requests Created Per Week")
                        fig_created = px.bar(
                            created_df,
                            x=chart_x_axis,
                            y='Count',
                            title="Service Requests Created Per Week",
                            labels={'Count': 'Number of SRs Created', chart_x_axis: 'Week Period'},
                            color_discrete_sequence=px.colors.qualitative.Plotly, # Optional: for a consistent color
                            text='Count' # Add text to display on bars
                        )
                        fig_created.update_traces(texttemplate='%{text}', textposition='outside') # Show text outside bars
                        fig_created.update_layout(xaxis_title='Week Period', yaxis_title="Number of SRs Created")
                        st.plotly_chart(fig_created, use_container_width=True)
                    else:
                        st.info("No data available for 'SRs Created Per Week' chart.")

                    if not closed_df.empty:
                        st.subheader("Service Requests Closed Per Week")
                        fig_closed = px.bar(
                            closed_df,
                            x=chart_x_axis,
                            y='Count',
                            title="Service Requests Closed Per Week: SR Status (Closed,Completed, Cancelled, Approval rejected, Rejected by ps)",
                            labels={'Count': 'Number of SRs Closed', chart_x_axis: 'Week Period'},
                            color_discrete_sequence=px.colors.qualitative.Plotly, # Optional: pick a different color if desired e.g., px.colors.qualitative.Plotly[1:]
                            text='Count' # Add text to display on bars
                        )
                        fig_closed.update_traces(texttemplate='%{text}', textposition='outside') # Show text outside bars
                        fig_closed.update_layout(xaxis_title='Week Period', yaxis_title="Number of SRs Closed")
                        st.plotly_chart(fig_closed, use_container_width=True)
                    else:
                        st.info("No data available for 'SRs Closed Per Week' chart.")

                st.markdown("---")
                st.subheader("Filterable SR Data")

                # Prepare data for table display and its filters
                table_display_df = sr_overview_df.copy() # This is the raw SR data for the table
                week_map_for_filter = {}
                week_options_for_multiselect = []

                # Populate week filter options from the combined data used for the chart
                if 'srs_weekly_combined_df' in locals() and not srs_weekly_combined_df.empty:
                    if 'WeekDisplay' in srs_weekly_combined_df.columns and 'Year-Week' in srs_weekly_combined_df.columns:
                        unique_week_options_df = srs_weekly_combined_df[['Year-Week', 'WeekDisplay']].drop_duplicates().sort_values(by='Year-Week')
                        week_options_for_multiselect = unique_week_options_df['WeekDisplay'].tolist()
                        for _, row in unique_week_options_df.iterrows():
                            week_map_for_filter[row['WeekDisplay']] = row['Year-Week']
                
                # The table_display_df needs 'Created On' and 'Year-Week' for filtering logic below
                if 'Created On' in table_display_df.columns:
                    table_display_df['Created On'] = pd.to_datetime(table_display_df['Created On'], errors='coerce')
                    # Keep rows with valid 'Created On' for the table, as filtering is based on this
                    table_display_df.dropna(subset=['Created On'], inplace=True) 
                    if not table_display_df.empty:
                         table_display_df['Year-Week'] = table_display_df['Created On'].dt.strftime('%G-W%V')
                else:
                    # If 'Created On' is not in table_display_df, week filtering on it won't work.
                    # Ensure 'Year-Week' column doesn't cause issues if it was expected.
                    if 'Year-Week' in table_display_df.columns: # Should not exist if 'Created On' didn't
                        pass # Or handle appropriately, e.g. disable week filter. For now, it will just be empty.

                col_filter1, col_filter2 = st.columns(2)

                with col_filter1:
                    selected_week_displays = st.multiselect(
                        "Filter by Week Period:",
                        options=week_options_for_multiselect,
                        default=[]
                    )

                with col_filter2:
                    # Corrected min_value and max_value for date_input
                    min_date_val = table_display_df['Created On'].min().date() if not table_display_df.empty and 'Created On' in table_display_df.columns and not table_display_df['Created On'].dropna().empty else None
                    max_date_val = table_display_df['Created On'].max().date() if not table_display_df.empty and 'Created On' in table_display_df.columns and not table_display_df['Created On'].dropna().empty else None
                    selected_day = st.date_input("Filter by Specific Day (Created On):", value=None, min_value=min_date_val, max_value=max_date_val)

                # Apply filters to table_display_df
                if selected_day:
                    table_display_df = table_display_df[table_display_df['Created On'].dt.date == selected_day]
                elif selected_week_displays:
                    selected_year_weeks_short = [week_map_for_filter[wd] for wd in selected_week_displays if wd in week_map_for_filter]
                    if selected_year_weeks_short:
                         table_display_df = table_display_df[table_display_df['Year-Week'].isin(selected_year_weeks_short)]

                # Display total row count using table_display_df
                st.markdown(f"**Total Displayed SRs:** {len(table_display_df)}")

                # Column selector using table_display_df
                if not table_display_df.empty:
                    all_columns = table_display_df.columns.tolist()
                    # Remove 'Year-Week' from selectable columns if it was added for filtering only
                    if 'Year-Week' in all_columns:
                        all_columns.remove('Year-Week')

                    default_cols = ['Service Request', 'Status', 'Created On']
                    sanitized_default_cols = [col for col in default_cols if col in all_columns]

                    if 'filterable_sr_data_cols_multiselect' not in st.session_state:
                        st.session_state.filterable_sr_data_cols_multiselect = sanitized_default_cols

                    selected_columns = st.multiselect(
                        "Select columns to display for Filterable SR Data:",
                        options=all_columns, # Offer all columns from the filtered DF
                        default=st.session_state.filterable_sr_data_cols_multiselect,
                        key="multiselect_filterable_sr_data"
                    )
                    st.session_state.filterable_sr_data_cols_multiselect = selected_columns


                    if selected_columns:
                        st.dataframe(table_display_df[selected_columns], hide_index=True)
                    else:
                        # Show all (minus internal Year-Week) if no columns are selected but data exists
                        st.dataframe(table_display_df[[col for col in all_columns if col != 'Year-Week']] if 'Year-Week' in all_columns else table_display_df, hide_index=True)
                else:
                    st.info("No SR data to display based on current filters for Filterable SR Data.")

                st.markdown("---") # Separator before the new Closed SRs table

                # --- Closed SRs Table ---
                st.subheader("Closed Service Requests")

                # Essential columns for this section
                essential_cols_closed_sr = ['Status', 'LastModDateTime']
                missing_essential_cols = [col for col in essential_cols_closed_sr if col not in sr_overview_df.columns]

                if missing_essential_cols:
                    st.warning(f"The uploaded SR data is missing the following essential column(s) for the Closed SRs table: {', '.join(missing_essential_cols)}. This table cannot be displayed.")
                else:
                    closed_sr_statuses = ["closed", "completed", "cancelled", "approval rejected", "rejected by ps"]
                    # Filter SRs that have one of the closed statuses
                    closed_srs_df = sr_overview_df[
                        sr_overview_df['Status'].astype(str).str.lower().str.strip().isin(closed_sr_statuses)
                    ].copy()

                    # Convert LastModDateTime to datetime and generate 'Closure-Year-Week'
                    closed_srs_df['LastModDateTime'] = pd.to_datetime(closed_srs_df['LastModDateTime'], errors='coerce', dayfirst=True, infer_datetime_format=True)
                    closed_srs_df.dropna(subset=['LastModDateTime'], inplace=True) # Remove rows where LastModDateTime couldn't be parsed

                    if not closed_srs_df.empty:
                        closed_srs_df['Closure-Year-Week'] = closed_srs_df['LastModDateTime'].dt.strftime('%G-W%V')
                    else:
                        # Ensure the column exists even if empty, for consistency in later steps
                        closed_srs_df['Closure-Year-Week'] = pd.Series(dtype='str')


                    # Prepare week filter options based on LastModDateTime of closed SRs
                    # This is different from the main week_options_for_multiselect which is based on Created On of all SRs
                    closed_sr_week_map_for_filter = {}
                    closed_sr_week_options_for_multiselect = []
                    if not closed_srs_df.empty and 'Closure-Year-Week' in closed_srs_df.columns:
                        unique_closed_week_options_df = closed_srs_df[['Closure-Year-Week']].copy()
                        unique_closed_week_options_df.dropna(subset=['Closure-Year-Week'], inplace=True)
                        # Apply the _get_week_display_str helper to the 'Closure-Year-Week'
                        # Ensure _get_week_display_str is available or define it if it's moved/not imported
                        # For now, assuming _get_week_display_str is accessible
                        unique_closed_week_options_df['WeekDisplay'] = unique_closed_week_options_df['Closure-Year-Week'].apply(_get_week_display_str)
                        unique_closed_week_options_df = unique_closed_week_options_df[['Closure-Year-Week', 'WeekDisplay']].drop_duplicates().sort_values(by='Closure-Year-Week')

                        closed_sr_week_options_for_multiselect = unique_closed_week_options_df['WeekDisplay'].tolist()
                        for _, row in unique_closed_week_options_df.iterrows():
                            closed_sr_week_map_for_filter[row['WeekDisplay']] = row['Closure-Year-Week']


                    col_filter_closed_sr1, col_filter_closed_sr2 = st.columns(2)

                    with col_filter_closed_sr1:
                        selected_week_displays_closed = st.multiselect(
                            "Filter Closed SRs by Closure Week Period:",
                            options=closed_sr_week_options_for_multiselect,
                            default=[],
                            key="closed_sr_closure_week_filter"
                        )

                    with col_filter_closed_sr2:
                        min_date_val_closed = closed_srs_df['LastModDateTime'].min().date() if not closed_srs_df.empty and not closed_srs_df['LastModDateTime'].dropna().empty else None
                        max_date_val_closed = closed_srs_df['LastModDateTime'].max().date() if not closed_srs_df.empty and not closed_srs_df['LastModDateTime'].dropna().empty else None
                        selected_day_closed = st.date_input(
                            "Filter Closed SRs by Specific Closure Day:",
                            value=None,
                            min_value=min_date_val_closed,
                            max_value=max_date_val_closed,
                            key="closed_sr_closure_day_filter"
                        )

                    # Apply filters to closed_srs_df
                    filtered_closed_srs_df = closed_srs_df.copy()

                    if selected_day_closed:
                        # Ensure LastModDateTime is date part for comparison
                        filtered_closed_srs_df = filtered_closed_srs_df[filtered_closed_srs_df['LastModDateTime'].dt.date == selected_day_closed]
                    elif selected_week_displays_closed:
                        if 'Closure-Year-Week' in filtered_closed_srs_df.columns and closed_sr_week_map_for_filter:
                            selected_closure_year_weeks_short = [closed_sr_week_map_for_filter[wd] for wd in selected_week_displays_closed if wd in closed_sr_week_map_for_filter]
                            if selected_closure_year_weeks_short:
                                filtered_closed_srs_df = filtered_closed_srs_df[filtered_closed_srs_df['Closure-Year-Week'].isin(selected_closure_year_weeks_short)]

                    st.markdown(f"**Total Displayed Closed SRs (filtered by closure date):** {len(filtered_closed_srs_df)}")

                    if not filtered_closed_srs_df.empty: # This is line 2108 approx.
                        all_closed_columns = filtered_closed_srs_df.columns.tolist() # Line 2109 - Start of block to indent
                        # Remove internal helper columns from user selection options
                        # 'Closure-Year-Week' is the one specifically created for this table's filtering.
                        # 'Year-Week' might exist if it was in the original sr_overview_df from created_on processing.
                        internal_cols_to_remove = ['Closure-Year-Week', 'Year-Week']
                        for col_to_remove in internal_cols_to_remove:
                            if col_to_remove in all_closed_columns:
                                all_closed_columns.remove(col_to_remove)

                        default_closed_cols = ['Service Request', 'Status', 'Created On', 'LastModDateTime', 'Resolution'] # Example, adjust as needed
                        sanitized_default_closed_cols = [col for col in default_closed_cols if col in all_closed_columns]

                        if 'closed_sr_data_cols_multiselect' not in st.session_state:
                            st.session_state.closed_sr_data_cols_multiselect = sanitized_default_closed_cols

                        selected_closed_columns = st.multiselect(
                            "Select columns to display for Closed SRs:",
                            options=all_columns,
                            default=st.session_state.closed_sr_data_cols_multiselect,
                            key="multiselect_closed_sr_data"
                        )
                        st.session_state.closed_sr_data_cols_multiselect = selected_closed_columns


                        if selected_closed_columns:
                            st.dataframe(filtered_closed_srs_df[selected_closed_columns], hide_index=True)
                        else:
                            # Show all available (minus internal Year-Week) if no columns are selected but data exists
                            # Ensure we use the correct list of all_closed_columns (which has helpers removed)
                            st.dataframe(filtered_closed_srs_df[all_closed_columns] if all_closed_columns else filtered_closed_srs_df, hide_index=True)

                        # Download button for Closed SRs
                        # Ensure download uses the correct set of columns (selected or all available for display)
                        cols_for_download = selected_closed_columns if selected_closed_columns else all_closed_columns
                        excel_closed_sr_data = generate_excel_download(filtered_closed_srs_df[cols_for_download] if cols_for_download else filtered_closed_srs_df)
                        st.download_button(
                            label="📥 Download Closed SRs Data",
                            data=excel_closed_sr_data,
                            file_name=f"closed_srs_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="download_closed_srs"
                        )
                    # This else corresponds to 'if not filtered_closed_srs_df.empty:'
                    else:
                        st.info("No Closed SR data to display based on current filters.")


st.markdown("---")
st.markdown(
    """<div style="text-align:center; color:#888; font-size:0.8em;">
    Intellipen SmartQ Test V4.0 | Developed by Ali Babiker | © July 2025
    </div>""",
    unsafe_allow_html=True
)
