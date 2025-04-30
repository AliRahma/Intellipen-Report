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
    h1, h2, h3 {
        color: #1565c0;
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
        options=["SR Analysis", "Not Resolved SR"],
        icons=["kanban", "clipboard-check"],
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
        important_cols = ['Last Note','Case Id', 'Current User Id', 'Case Start Date', 'Status', 'Type', 'Ticket Number', 'Age (Days)']
        
        # Add SR Status columns if available
        if 'SR Status' in df_display.columns:
            important_cols.extend(['SR Status', 'Last Update'])
        
        # Ensure all columns exist
        display_cols = [col for col in important_cols if col in df_display.columns]
        
        # Add checkbox column for tracking
        df_display_with_checkbox = df_display[display_cols].copy()
        
        # Create a dataframe for display
        if not df_display_with_checkbox.empty:
            # Add an action column for tracking
            df_display_with_checkbox['Track'] = [
                "✓" if row['Case Id'] in [tracked['Case Id'] for tracked in st.session_state.tracked_rows] else ""
                for _, row in df_display_with_checkbox.iterrows()
            ]
            
            # Display the table
            st.dataframe(df_display_with_checkbox)
            
            # Add tracking functionality
            st.subheader("🔍 Track Cases")
            track_case_id = st.selectbox(
                "Select a case to track/untrack:",
                df_display['Case Id'].tolist()
            )
            
            if track_case_id and st.button("Toggle Tracking"):
                case_row = df_display[df_display['Case Id'] == track_case_id].iloc[0]
                track_row(case_row)
                st.success(f"Case {track_case_id} tracking status toggled!")
        
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
            
            # Display tracked items in a table
            st.subheader("📋 Tracked Items List")
            
            # Prepare display columns
            display_cols = ['Case Id', 'Current User Id', 'Case Start Date', 'Status', 'Type', 'Ticket Number', 'Age (Days)']
            
            # Add SR Status columns if available
            if 'SR Status' in tracked_df.columns:
                display_cols.extend(['SR Status', 'Last Update'])
            
            # Ensure all columns exist
            final_cols = [col for col in display_cols if col in tracked_df.columns]
            
            # Display the tracking table
            st.dataframe(tracked_df[final_cols])
            
            # Add functionality to remove tracked items
            st.subheader("🗑️ Remove Tracked Item")
            
            remove_case_id = st.selectbox(
                "Select a case to remove from tracking:",
                tracked_df['Case Id'].tolist()
            )
            
            if remove_case_id and st.button("Remove from Tracking"):
                case_row = tracked_df[tracked_df['Case Id'] == remove_case_id].iloc[0]
                track_row(case_row)
                st.success(f"Case {remove_case_id} removed from tracking!")

# Run the application
if __name__ == "__main__":
    pass