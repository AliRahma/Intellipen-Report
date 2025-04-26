import streamlit as st
import pandas as pd
import re
import io
import datetime

# File paths (to store Last Updated timestamp and PIN code)
LAST_UPDATED_FILE = "last_updated.txt"
PIN_CODE = "1234"  # Replace with your own secure PIN code

# Helper function to read/write the last updated timestamp
def get_last_updated():
    try:
        with open(LAST_UPDATED_FILE, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        return "Never updated"

def set_last_updated():
    with open(LAST_UPDATED_FILE, "w") as file:
        file.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Streamlit page configuration
st.set_page_config(page_title="SR Dashboard", layout="wide")

# Sidebar: Display last updated timestamp
st.sidebar.write(f"**Last Updated:** {get_last_updated()}")

# Sidebar: Input PIN code for uploading files
st.sidebar.header("🔒 Secure Update")
pin_input = st.sidebar.text_input("Enter PIN code to update", type="password")

# Sidebar: Upload files
uploaded_file = None
if pin_input == PIN_CODE:
    st.sidebar.success("PIN code is correct! You can upload files.")
    uploaded_file = st.sidebar.file_uploader("📂 Upload Main Excel File (.xlsx)", type="xlsx")
else:
    st.sidebar.warning("Enter a valid PIN code to enable file uploads.")

sr_status_file = st.sidebar.file_uploader("📂 Upload SR Status Excel (optional)", type="xlsx")

# Initialize filter variable
sr_status_filter = None

# Process uploaded file
if uploaded_file:
    try:
        # Update the "Last Updated" timestamp
        set_last_updated()

        # Load the main Excel file
        df = pd.read_excel(uploaded_file)

        # Column setup
        user_col = 'Current User Id'
        note_col = 'Last Note'
        date_col = 'Case Start Date'
        target_users = ['ali.babiker', 'anas.hasan', 'ahmed.mostafa']
        df_filtered = df[df[user_col].isin(target_users)].copy()
        df_filtered[date_col] = pd.to_datetime(df_filtered[date_col], errors='coerce')

        # Classification logic
        def classify_and_extract(note):
            if not isinstance(note, str):
                return "Not Triaged", None, None
            note_lower = note.lower()
            match = re.search(r'(tkt|sr|inc|ticket|مرجعي|incident|اس ار|انسدنت)[\s\S]{0,50}?(\d{4,})', note_lower)
            if match:
                ticket_num = int(match.group(2))
                return "Pending SR/Incident", ticket_num, "SR" if 14000 <= ticket_num <= 16000 else "Incident"
            return "Not Triaged", None, None

        df_filtered[['Status', 'Ticket Number', 'Type']] = df_filtered[note_col].apply(
            lambda x: pd.Series(classify_and_extract(x))
        )

        # Merge SR status if uploaded
        if sr_status_file:
            try:
                sr_df = pd.read_excel(sr_status_file)
                sr_df['Service Request'] = sr_df['Service Request'].astype(str).str.extract(r'(\d{4,})')
                sr_df['Service Request'] = sr_df['Service Request'].astype(float).astype("Int64")

                sr_df = sr_df.rename(columns={
                    'Status': 'SR Status',
                    'LastModDateTime': 'Last Update'
                })

                df_filtered['Ticket Number'] = df_filtered['Ticket Number'].astype("Int64")
                df_filtered = df_filtered.merge(
                    sr_df[['Service Request', 'SR Status', 'Last Update']],
                    how='left',
                    left_on='Ticket Number',
                    right_on='Service Request'
                ).drop(columns=['Service Request'])

                # Add SR Status filter to sidebar
                st.sidebar.markdown("---")
                sr_status_options = df_filtered['SR Status'].fillna("None").unique().tolist()
                sr_status_filter = st.sidebar.selectbox("📌 Filter by SR Status", ["All"] + sr_status_options)

            except Exception as e:
                st.error(f"Error merging SR Status file: {e}")

        # Other sidebar filters
        st.sidebar.markdown("---")
        status_filter = st.sidebar.selectbox("📌 Filter by Triage Status", ["All"] + df_filtered["Status"].dropna().unique().tolist())
        type_filter = st.sidebar.selectbox("📌 Filter by Type", ["All", "SR", "Incident"])

        df_display = df_filtered.copy()
        if status_filter != "All":
            df_display = df_display[df_display["Status"] == status_filter]
        if type_filter != "All":
            df_display = df_display[df_display["Type"] == type_filter]
        if sr_status_filter and sr_status_filter != "All":
            if sr_status_filter == "None":
                df_display = df_display[df_display["SR Status"].isna()]
            else:
                df_display = df_display[df_display["SR Status"] == sr_status_filter]

        # Search
        st.subheader("🔎 Search for Ticket Number")
        search_input = st.text_input("Enter SR or Incident Number (e.g., 15023):")
        if search_input.isdigit():
            search_number = int(search_input)
            df_display = df_display[df_display['Ticket Number'] == search_number]

        # Display filtered results
        st.subheader("📋 Filtered Results")
        st.markdown(f"**Total Filtered Rows:** {df_display.shape[0]}")
        st.dataframe(df_display)

        # Excel download
        def generate_excel_download(data):
            output = io.BytesIO()
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            data.to_excel(writer, index=False, sheet_name='Results')
            writer.close()
            output.seek(0)
            return output

        excel_data = generate_excel_download(df_display)
        st.download_button(
            label="📥 Download Filtered Data to Excel",
            data=excel_data,
            file_name="filtered_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"Something went wrong: {e}")
