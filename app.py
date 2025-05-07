import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import base64
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(
    page_title="SR Analyzer Pro Test",
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
# Session state for tracked rows
if 'tracked_rows' not in st.session_state:
    st.session_state.tracked_rows = []
# Session state for selected rows
if 'selected_case_ids' not in st.session_state:
    st.session_state.selected_case_ids = []

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

# Function to handle bulk tracking operations
def bulk_track_toggle(df, case_ids):
    """Toggle tracking status for multiple cases at once"""
    if not case_ids:
        return
        
    for case_id in case_ids:
        # Get the row data for this case
        case_data = df[df['Case Id'] == case_id].iloc[0]
        track_row(case_data)
    
    # Clear selection after processing
    st.session_state.selected_case_ids = []

# Sidebar - File Upload Section
with st.sidebar:
    st.title("📊 SR Analyzer Pro")
    st.markdown("---")

    st.subheader("📁 Data Import")
    uploaded_file = st.file_uploader("Upload Main Excel File (.xlsx)", type=["xlsx","xls"])
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
    - Track unresolved SRs
    - Multi-select tracking
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
        options=["SR Analysis", "Not Resolved SR", "Today's SR/Incidents"],
        icons=["kanban", "clipboard-check", "calendar-date"],
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
                sr_df[['Service Request', 'SR Status', 'Last Update']],
                how='left',
                left_on='Ticket Number',
                right_on='Service Request'
            ).drop(columns=['Service Request'])
        
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
        
        # Display data table with important columns and multi-select functionality
        important_cols = ['Last Note', 'Case Id', 'Current User Id', 'Case Start Date', 'Status', 'Type', 'Ticket Number']
        
        # Add SR Status columns if available
        if 'SR Status' in df_display.columns:
            important_cols.extend(['SR Status', 'Last Update'])
        
        # Ensure all columns exist
        display_cols = [col for col in important_cols if col in df_display.columns]
        
        # Prepare dataframe with selection column and tracking status
        if not df_display.empty:
            # Create a new dataframe for display
            df_selection = df_display[display_cols].copy()
            
            # Add tracking status column
            df_selection['Tracked'] = [
                "✓" if row['Case Id'] in [tracked['Case Id'] for tracked in st.session_state.tracked_rows] else ""
                for _, row in df_selection.iterrows()
            ]
            
            # Convert to a list of dictionaries for the data editor
            selection_data = df_selection.to_dict('records')
            
            # Create multi-select dataframe
            st.markdown("Select rows below to track/untrack them:")
            
            # Display with st.data_editor for selection functionality
            edited_df = st.data_editor(
                df_selection,
                column_config={
                    "Tracked": st.column_config.CheckboxColumn(
                        "Track",
                        help="Select to track/untrack",
                        default=False,
                    )
                },
                hide_index=True,
                key="selection_table"
            )
            
            # Handle the tracking toggle for selected rows
            if st.button("Toggle Tracking for Selected Rows", key="toggle_tracking"):
                # Get rows where 'Tracked' is checked
                selected_rows = edited_df[edited_df['Tracked'] == True]
                
                # Process each selected row
                if not selected_rows.empty:
                    for _, row in selected_rows.iterrows():
                        case_id = row['Case Id']
                        case_data = df_display[df_display['Case Id'] == case_id].iloc[0]
                        track_row(case_data)
                    
                    st.success(f"Toggled tracking for {len(selected_rows)} items!")
                    st.rerun()
                else:
                    st.warning("No rows selected. Please select rows to track/untrack.")
            
            # Button to track all visible rows
            if st.button("Track All Visible Rows", key="track_all"):
                # Get all visible case IDs
                visible_case_ids = df_display['Case Id'].tolist()
                
                # Add tracking for all visible cases
                for case_id in visible_case_ids:
                    # Check if not already tracked
                    if case_id not in [row['Case Id'] for row in st.session_state.tracked_rows]:
                        case_data = df_display[df_display['Case Id'] == case_id].iloc[0]
                        track_row(case_data)
                
                st.success(f"Added {len(visible_case_ids)} items to tracking!")
                st.rerun()
            
            # Button to un-track all visible rows
            if st.button("Untrack All Visible Rows", key="untrack_all"):
                # Get all visible case IDs
                visible_case_ids = df_display['Case Id'].tolist()
                
                # Count items to untrack
                to_untrack = [case_id for case_id in visible_case_ids if case_id in [row['Case Id'] for row in st.session_state.tracked_rows]]
                
                # Remove tracking for all visible cases
                st.session_state.tracked_rows = [
                    row for row in st.session_state.tracked_rows 
                    if row['Case Id'] not in visible_case_ids
                ]
                
                if to_untrack:
                    st.success(f"Removed {len(to_untrack)} items from tracking!")
                    st.rerun()
                else:
                    st.info("No tracked items found in the current view.")
        
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
            
            # Display as a table
            st.table(pd.DataFrame(case_details))
            
            # Track/untrack button for this specific case
            is_tracked = case_row['Case Id'] in [row['Case Id'] for row in st.session_state.tracked_rows]
            track_button_label = "🔄 Remove from Tracking" if is_tracked else "✅ Add to Tracking"
            
            if st.button(track_button_label):
                track_row(case_row)
                st.success(f"Case {case_row['Case Id']} {'removed from' if is_tracked else 'added to'} tracking!")
                st.rerun()
            
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
            
            # Display statistics in a table
            st.subheader("📊 Tracked SR Statistics")
            
            # Create statistics table
            stats_dict = {
                "Metric": ["Total Tracked Items", "Service Requests", "Incidents"],
                "Count": [
                    len(tracked_df),
                    len(tracked_df[tracked_df['Type'] == 'SR']),
                    len(tracked_df[tracked_df['Type'] == 'Incident'])
                ]
            }
            
            st.table(pd.DataFrame(stats_dict))
            
            # Download button
            st.download_button(
                label="📥 Download Tracked Items",
                data=generate_excel_download(tracked_df),
                file_name=f"tracked_items_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            # Add batch untrack functionality
            st.subheader("📋 Tracked Items List")
            
            # Prepare display columns
            display_cols = ['Case Id', 'Current User Id', 'Case Start Date', 'Status', 'Type', 'Ticket Number', 'Age (Days)']
            
            # Add SR Status columns if available
            if 'SR Status' in tracked_df.columns:
                display_cols.extend(['SR Status', 'Last Update'])
            
            # Filter columns that exist in the dataframe
            display_cols = [col for col in display_cols if col in tracked_df.columns]
            
            # Display the tracked items with checkboxes for multi-select
            edited_tracked_df = st.data_editor(
                tracked_df[display_cols],
                column_config={
                    "Select": st.column_config.CheckboxColumn(
                        "Select",
                        help="Select to remove from tracking",
                        default=False,
                    )
                },
                hide_index=True,
                key="tracked_items_table"
            )
            
            # Button to untrack selected items
            if st.button("Untrack Selected Items", key="untrack_selected"):
                # Get the selected rows from the edited dataframe
                if "Select" in edited_tracked_df.columns:
                    selected_to_untrack = edited_tracked_df[edited_tracked_df["Select"] == True]['Case Id'].tolist()
                    
                    if selected_to_untrack:
                        # Remove selected rows from tracked rows
                        st.session_state.tracked_rows = [
                            row for row in st.session_state.tracked_rows 
                            if row['Case Id'] not in selected_to_untrack
                        ]
                        
                        st.success(f"Removed {len(selected_to_untrack)} items from tracking!")
                        st.rerun()
                    else:
                        st.warning("No items selected for untracking.")
            
            # Clear all tracked items button
            if st.button("Clear All Tracked Items", key="clear_tracked"):
                st.session_state.tracked_rows = []
                st.success("All tracked items cleared!")
                st.rerun()
            
            # Detail view of a selected tracked item
            st.subheader("🔍 Tracked Item Details")
            
            if not tracked_df.empty:
                selected_tracked_case = st.selectbox(
                    "Select a tracked case to view details:",
                    tracked_df['Case Id'].tolist(),
                    key="tracked_case_select"
                )
                
                if selected_tracked_case:
                    tracked_case_row = tracked_df[tracked_df['Case Id'] == selected_tracked_case].iloc[0]
                    
                    # Create columns for layout
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Case details
                        st.markdown("### Case Information")
                        
                        # Prepare case details
                        details = {
                            "Field": ["Case ID", "Owner", "Start Date", "Age"],
                            "Value": [
                                tracked_case_row['Case Id'],
                                tracked_case_row['Current User Id'],
                                tracked_case_row['Case Start Date'].strftime('%Y-%m-%d') if 'Case Start Date' in tracked_case_row and not pd.isna(tracked_case_row['Case Start Date']) else 'N/A',
                                f"{tracked_case_row['Age (Days)']} days" if 'Age (Days)' in tracked_case_row and not pd.isna(tracked_case_row['Age (Days)']) else 'N/A'
                            ]
                        }
                        
                        # Add SR-related details
                        if 'Ticket Number' in tracked_case_row and not pd.isna(tracked_case_row['Ticket Number']):
                            details["Field"].extend(["Ticket Number", "Type"])
                            details["Value"].extend([
                                int(tracked_case_row['Ticket Number']),
                                tracked_case_row['Type'] if not pd.isna(tracked_case_row['Type']) else 'N/A'
                            ])
                        
                        # Add SR status if available
                        if 'SR Status' in tracked_case_row and not pd.isna(tracked_case_row['SR Status']):
                            details["Field"].append("SR Status")
                            details["Value"].append(tracked_case_row['SR Status'])
                            
                            if 'Last Update' in tracked_case_row and not pd.isna(tracked_case_row['Last Update']):
                                details["Field"].append("Last Update")
                                details["Value"].append(tracked_case_row['Last Update'])
                        
                        # Show case details table
                        st.table(pd.DataFrame(details))
                    
                    with col2:
                        # Note content
                        st.markdown("### Last Note")
                        
                        if 'Last Note' in tracked_case_row and not pd.isna(tracked_case_row['Last Note']):
                            st.text_area("Note Content", tracked_case_row['Last Note'], height=250, key="tracked_note")
                        else:
                            st.info("No notes available for this case")
                    
                    # Button to remove this specific tracked item
                    if st.button("Remove from Tracking", key="remove_single_tracked"):
                        # Remove from tracked rows
                        st.session_state.tracked_rows = [
                            row for row in st.session_state.tracked_rows 
                            if row['Case Id'] != selected_tracked_case
                        ]
                        
                        st.success(f"Removed case {selected_tracked_case} from tracking!")
                        st.rerun()
    
    #
    # TODAY'S SR/INCIDENTS TAB
    #
    elif selected == "Today's SR/Incidents":
        st.title("📆 Today's New SR/Incidents")
        
        # Get all items created today
        df_today = df_enriched[df_enriched['Created Today'] == True].copy()
        
        # Display summary statistics
        st.subheader("📊 Today's Activity Summary")
        
        # Statistics in cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            sr_count = len(df_today[df_today['Type'] == 'SR'])
            st.markdown(f'<p class="metric-value">{sr_count}</p>', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">New Service Requests Today</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            incident_count = len(df_today[df_today['Type'] == 'Incident'])
            st.markdown(f'<p class="metric-value">{incident_count}</p>', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">New Incidents Today</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            total_count = len(df_today)
            st.markdown(f'<p class="metric-value">{total_count}</p>', unsafe_allow_html=True)
            st.markdown('<p class="metric-label">Total New Items Today</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display today's items
        st.subheader("📋 Today's New Items")
        
        if df_today.empty:
            st.info("No new SR/Incidents found for today.")
        else:
            # Filter options
            today_filter_col1, today_filter_col2 = st.columns(2)
            
            with today_filter_col1:
                today_type_filter = st.selectbox(
                    "Filter by Type",
                    ["All", "SR", "Incident"],
                    key="today_type_filter"
                )
            
            with today_filter_col2:
                today_user_filter = st.selectbox(
                    "Filter by User",
                    ["All"] + df_today['Current User Id'].dropna().unique().tolist(),
                    key="today_user_filter"
                )
            
            # Apply filters
            df_today_filtered = df_today.copy()
            
            if today_type_filter != "All":
                df_today_filtered = df_today_filtered[df_today_filtered['Type'] == today_type_filter]
            
            if today_user_filter != "All":
                df_today_filtered = df_today_filtered[df_today_filtered['Current User Id'] == today_user_filter]
            
            # Display filtered results
            st.markdown(f"**Filtered Results:** {len(df_today_filtered)} items")
            
            # Download button
            if not df_today_filtered.empty:
                st.download_button(
                    label="📥 Download Today's Items",
                    data=generate_excel_download(df_today_filtered),
                    file_name=f"today_items_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            # Display columns
            display_cols = ['Case Id', 'Current User Id', 'Case Start Date', 'Status', 'Type', 'Ticket Number']
            
            # Include SR Status if available
            if 'SR Status' in df_today_filtered.columns:
                display_cols.extend(['SR Status', 'Last Update'])
            
            # Filter columns that exist in the dataframe
            display_cols = [col for col in display_cols if col in df_today_filtered.columns]
            
            # Display the data
            st.dataframe(df_today_filtered[display_cols], hide_index=True)
            
            # Add tracking functionality
            st.subheader("🔍 Track Today's Items")
            
            # Select items to track
            selected_case_today = st.selectbox(
                "Select a case to view details or track:",
                df_today_filtered['Case Id'].tolist(),
                key="today_case_select"
            )
            
            if selected_case_today:
                today_case_row = df_today_filtered[df_today_filtered['Case Id'] == selected_case_today].iloc[0]
                
                # Display case details
                st.markdown("### Case Details")
                
                # Create columns for layout
                col1, col2 = st.columns(2)
                
                with col1:
                    # Case information
                    details = {
                        "Field": ["Case ID", "Owner", "Start Date"],
                        "Value": [
                            today_case_row['Case Id'],
                            today_case_row['Current User Id'],
                            today_case_row['Case Start Date'].strftime('%Y-%m-%d') if not pd.isna(today_case_row['Case Start Date']) else 'N/A'
                        ]
                    }
                    
                    # Add SR-related details
                    if 'Ticket Number' in today_case_row and not pd.isna(today_case_row['Ticket Number']):
                        details["Field"].extend(["Ticket Number", "Type"])
                        details["Value"].extend([
                            int(today_case_row['Ticket Number']),
                            today_case_row['Type'] if not pd.isna(today_case_row['Type']) else 'N/A'
                        ])
                    
                    # Add SR status if available
                    if 'SR Status' in today_case_row and not pd.isna(today_case_row['SR Status']):
                        details["Field"].append("SR Status")
                        details["Value"].append(today_case_row['SR Status'])
                        
                        if 'Last Update' in today_case_row and not pd.isna(today_case_row['Last Update']):
                            details["Field"].append("Last Update")
                            details["Value"].append(today_case_row['Last Update'])
                    
                    # Show case details table
                    st.table(pd.DataFrame(details))
                
                with col2:
                    # Note content
                    st.markdown("### Last Note")
                    
                    if 'Last Note' in today_case_row and not pd.isna(today_case_row['Last Note']):
                        st.text_area("Note Content", today_case_row['Last Note'], height=250, key="today_note")
                    else:
                        st.info("No notes available for this case")
                
                # Check if item is already tracked
                is_tracked = today_case_row['Case Id'] in [row['Case Id'] for row in st.session_state.tracked_rows]
                
                # Track/untrack button
                track_button_label = "🔄 Remove from Tracking" if is_tracked else "✅ Add to Tracking"
                
                if st.button(track_button_label, key="today_track_button"):
                    track_row(today_case_row)
                    st.success(f"Case {today_case_row['Case Id']} {'removed from' if is_tracked else 'added to'} tracking!")
                    st.rerun()
            
            # Add multi-select tracking functionality
            st.subheader("📋 Bulk Track Today's Items")
            
            # Create dataframe with selection column
            df_today_select = df_today_filtered[display_cols].copy()
            
            # Add tracking status column
            df_today_select['Tracked'] = [
                "✓" if row['Case Id'] in [tracked['Case Id'] for tracked in st.session_state.tracked_rows] else ""
                for _, row in df_today_select.iterrows()
            ]
            
            # Display with selection capability
            edited_today_df = st.data_editor(
                df_today_select,
                column_config={
                    "Select": st.column_config.CheckboxColumn(
                        "Select",
                        help="Select to track these items",
                        default=False,
                    )
                },
                hide_index=True,
                key="today_selection_table"
            )
            
            # Button to track selected items
            if st.button("Track Selected Items", key="today_track_selected"):
                # Check if Select column exists
                if "Select" in edited_today_df.columns:
                    selected_to_track = edited_today_df[edited_today_df["Select"] == True]['Case Id'].tolist()
                    
                    if selected_to_track:
                        # Track each selected item
                        for case_id in selected_to_track:
                            # Check if not already tracked
                            if case_id not in [row['Case Id'] for row in st.session_state.tracked_rows]:
                                case_data = df_today_filtered[df_today_filtered['Case Id'] == case_id].iloc[0]
                                track_row(case_data)
                        
                        st.success(f"Added {len(selected_to_track)} items to tracking!")
                        st.rerun()
                    else:
                        st.warning("No items selected for tracking.")
            
            # Button to track all today's items
            if st.button("Track All Today's Items", key="track_all_today"):
                # Get all case IDs
                all_today_cases = df_today_filtered['Case Id'].tolist()
                
                # Count new items to track
                new_to_track = [case_id for case_id in all_today_cases if case_id not in [row['Case Id'] for row in st.session_state.tracked_rows]]
                
                # Track each item
                for case_id in new_to_track:
                    case_data = df_today_filtered[df_today_filtered['Case Id'] == case_id].iloc[0]
                    track_row(case_data)
                
                if new_to_track:
                    st.success(f"Added {len(new_to_track)} new items to tracking!")
                    st.rerun()
                else:
                    st.info("All items are already being tracked.")

# Run the app
if __name__ == "__main__":
    pass  # The Streamlit script is run directly when imported