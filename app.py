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

# Prepare tab interface - with removed tabs
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

#
# SR ANALYSIS TAB
#
if selected == "SR Analysis":
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

# Run the application
if __name__ == "__main__":
    pass