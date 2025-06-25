import pandas as pd
import io
from datetime import datetime, timedelta # Added timedelta
import numpy as np

# Function to classify and extract ticket info
def classify_and_extract(note, ticket_regex, sr_min_range, sr_max_range):
    if not isinstance(note, str):
        return "Not Triaged", None, None

    note_lower = note.lower()
    try:
        import re # Ensure re is imported if not globally available
        match = re.search(ticket_regex, note_lower, re.IGNORECASE | re.DOTALL)
    except Exception as e: # Catch issues with regex compilation
        # st.error(f"Regex error: {e}") # Would need st here, better to log
        print(f"Regex error: {e}")
        return "Regex Error", None, None

    if match:
        try:
            ticket_num_str = match.group(2) # Assuming group 2 is the number
            if ticket_num_str:
                ticket_num = int(ticket_num_str)
                ticket_type = "SR" if sr_min_range <= ticket_num <= sr_max_range else "Incident"
                return "Pending SR/Incident", ticket_num, ticket_type
        except (IndexError, ValueError):
            # If group 2 doesn't exist or not a number, it's not a valid ticket format
            return "Not Triaged", None, None

    return "Not Triaged", None, None

# Function to calculate case age in days
def calculate_age(start_date):
    if pd.isna(start_date) or not isinstance(start_date, datetime):
        return None
    return (datetime.now() - start_date).days

# Function to determine if a note was created today
def is_created_today(date_value):
    if pd.isna(date_value):
        return False
    today = datetime.now().date()
    try:
        note_date = date_value.date() if isinstance(date_value, datetime) else pd.to_datetime(date_value).date()
    except: # Handle cases where date_value might not be directly convertible
        return False
    return note_date == today

# Function to create downloadable Excel
def generate_excel_download(data, sheet_name='Results'):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        data.to_excel(writer, index=False, sheet_name=sheet_name)
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#1976d2',
            'color': 'white',
            'border': 1,
            'text_wrap': True,
            'valign': 'vcenter',
            'align': 'center'
        })

        for col_num, value in enumerate(data.columns.values):
            worksheet.write(0, col_num, value, header_format)

        for i, col in enumerate(data.columns):
            column_width = max(data[col].astype(str).apply(len).max(), len(str(col))) + 2
            if column_width > 50: # Cap max width
                column_width = 50
            worksheet.set_column(i, i, column_width)
    output.seek(0)
    return output

# Function to create downloadable CSV
def generate_csv_download(data):
    output = io.StringIO()
    data.to_csv(output, index=False)
    output.seek(0)
    return output.getvalue()

def time_since_breach(breach_date, resolution_date=None):
    if pd.isna(breach_date):
        return None

    breach_dt = pd.to_datetime(breach_date, errors='coerce')
    if pd.isna(breach_dt):
        return None

    if resolution_date and not pd.isna(resolution_date):
        end_dt = pd.to_datetime(resolution_date, errors='coerce')
        if pd.isna(end_dt):
            return "Invalid Resolution Date"
    else:
        end_dt = datetime.now()

    delta = end_dt - breach_dt
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, _ = divmod(remainder, 60)

    if days < 0: # Breach hasn't happened yet or resolution before breach (data issue)
        return "Breach Not Reached / Resolved Before"

    return f"{days}d {hours}h {minutes}m"

def time_to_resolve_after_breach(breach_date, resolution_date):
    if pd.isna(breach_date) or pd.isna(resolution_date):
        return None

    breach_dt = pd.to_datetime(breach_date, errors='coerce')
    res_dt = pd.to_datetime(resolution_date, errors='coerce')

    if pd.isna(breach_dt) or pd.isna(res_dt) or res_dt < breach_dt:
        return None # Or "Resolved before breach / Invalid dates"

    delta = res_dt - breach_dt
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{days}d {hours}h {minutes}m"

def calculate_team_status_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the summary of incidents grouped by Team and Status.

    Args:
        df: Input DataFrame, expected to have 'Team' and 'Status' columns.

    Returns:
        A DataFrame with columns ['Team', 'Status', 'Total Incidents']
        summarizing the count of incidents. Returns an empty DataFrame
        with these columns if 'Team' or 'Status' is missing in the input.
    """
    if 'Team' in df.columns and 'Status' in df.columns:
        summary_df = df.groupby(['Team', 'Status']).size().reset_index(name='Total Incidents')
    else:
        summary_df = pd.DataFrame(columns=['Team', 'Status', 'Total Incidents'])
    return summary_df

def test_case_count_calculation_and_filtering():
    """Tests for Case Count calculation and linked cases filtering logic."""
    print("Running test_case_count_calculation_and_filtering...")

    # 1. Test Case Count Calculation
    print("  Testing Case Count Calculation...")
    case_count_data = {
        'Ticket Number': ['INC100', 'SR200', 'INC100', 'SR300', 'INC100', 'SR200'],
        'Type': ['Incident', 'SR', 'Incident', 'SR', 'Incident', 'SR'],
        'OtherData': [1, 2, 3, 4, 5, 6]
    }
    df_case_count_test = pd.DataFrame(case_count_data)
    df_case_count_test['Case Count'] = df_case_count_test.groupby(['Ticket Number', 'Type'])['Ticket Number'].transform('size')

    expected_case_counts = pd.Series([3, 2, 3, 1, 3, 2], name='Case Count')
    pd.testing.assert_series_equal(df_case_count_test['Case Count'], expected_case_counts, check_dtype=False)
    print("  Case Count Calculation Test Passed.")

    # 2. Test Filtering Logic for Linked Cases
    print("  Testing Linked Cases Filtering Logic...")
    filtering_data = {
        'Ticket Number': ['INC001', 'SR002', 'INC003', 'SR004', 'INC001', None, 'SR005', 'SR002'],
        'Type': ['Incident', 'SR', 'Incident', 'SR', 'Incident', 'SR', 'SR', 'SR'],
        'Case Count': [3, 2, 1, 1, 3, 2, 4, 2],
        'Details': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    }
    df_filtering_test = pd.DataFrame(filtering_data)
    min_linked_cases = 2

    # Apply filtering as done in app.py
    linked_cases_df = df_filtering_test[
        (df_filtering_test['Case Count'] >= min_linked_cases) &
        (df_filtering_test['Ticket Number'].notna())
    ]

    # Create summary as done in app.py
    if not linked_cases_df.empty:
        linked_summary_df = linked_cases_df[['Ticket Number', 'Type', 'Case Count']].drop_duplicates().sort_values(
            by='Case Count', ascending=False
        ).reset_index(drop=True)
    else:
        linked_summary_df = pd.DataFrame(columns=['Ticket Number', 'Type', 'Case Count'])

    expected_summary_data = {
        'Ticket Number': ['SR005', 'INC001', 'SR002'],
        'Type': ['SR', 'Incident', 'SR'],
        'Case Count': [4, 3, 2]
    }
    df_expected_summary = pd.DataFrame(expected_summary_data)

    pd.testing.assert_frame_equal(linked_summary_df, df_expected_summary, check_dtype=False)
    print("  Linked Cases Filtering Logic Test Passed.")

    # Test Filtering Logic - Edge case: No items meet criteria
    print("  Testing Linked Cases Filtering Logic (Edge Case: No items)...")
    min_linked_cases_high = 5
    linked_cases_df_edge = df_filtering_test[
        (df_filtering_test['Case Count'] >= min_linked_cases_high) &
        (df_filtering_test['Ticket Number'].notna())
    ]
    if not linked_cases_df_edge.empty:
        linked_summary_df_edge = linked_cases_df_edge[['Ticket Number', 'Type', 'Case Count']].drop_duplicates().sort_values(
            by='Case Count', ascending=False
        ).reset_index(drop=True)
    else:
        linked_summary_df_edge = pd.DataFrame(columns=['Ticket Number', 'Type', 'Case Count'])

    df_expected_empty_summary = pd.DataFrame(columns=['Ticket Number', 'Type', 'Case Count'])
    pd.testing.assert_frame_equal(linked_summary_df_edge, df_expected_empty_summary, check_dtype=False)
    print("  Linked Cases Filtering Logic (Edge Case: No items) Test Passed.")

    print("All test_case_count_calculation_and_filtering tests passed.")

def test_calculate_team_status_summary():
    """Tests for the calculate_team_status_summary function."""
    print("Running test_calculate_team_status_summary...")
    # Test with valid data
    sample_data = {
        'Team': ['Alpha', 'Alpha', 'Bravo', 'Alpha', 'Bravo', 'Charlie'],
        'Status': ['Open', 'Closed', 'Open', 'Open', 'In Progress', 'Open'],
        'ID': [1, 2, 3, 4, 5, 6]
    }
    test_df = pd.DataFrame(sample_data)
    summary = calculate_team_status_summary(test_df)

    assert not summary.empty, "Test Case 1 Failed: Summary should not be empty for valid data."
    assert summary.shape == (5, 3), f"Test Case 1 Failed: Expected shape (5, 3), got {summary.shape}"

    # Check a specific row: Alpha, Open should be 2
    alpha_open_count_series = summary[(summary['Team'] == 'Alpha') & (summary['Status'] == 'Open')]['Total Incidents']
    assert not alpha_open_count_series.empty, "Test Case 1 Failed: 'Alpha' team with 'Open' status not found."
    alpha_open_count = alpha_open_count_series.iloc[0]
    assert alpha_open_count == 2, f"Test Case 1 Failed: Expected Alpha-Open count 2, got {alpha_open_count}"

    # Check another specific row: Bravo, In Progress should be 1
    bravo_in_progress_series = summary[(summary['Team'] == 'Bravo') & (summary['Status'] == 'In Progress')]['Total Incidents']
    assert not bravo_in_progress_series.empty, "Test Case 1 Failed: 'Bravo' team with 'In Progress' status not found."
    bravo_in_progress_count = bravo_in_progress_series.iloc[0]
    assert bravo_in_progress_count == 1, f"Test Case 1 Failed: Expected Bravo-In Progress count 1, got {bravo_in_progress_count}"

    print("Test Case 1 (Valid Data) Passed.")

    # Test with missing columns
    sample_data_missing_cols = {
        'Group': ['A', 'B'],
        'Value': [10, 20]
    }
    test_df_missing = pd.DataFrame(sample_data_missing_cols)
    summary_missing = calculate_team_status_summary(test_df_missing)

    assert summary_missing.empty, "Test Case 2 Failed: Summary should be empty when columns are missing."
    expected_cols = ['Team', 'Status', 'Total Incidents']
    assert list(summary_missing.columns) == expected_cols, \
        f"Test Case 2 Failed: Expected columns {expected_cols}, got {list(summary_missing.columns)}"
    print("Test Case 2 (Missing Columns) Passed.")

    # Test with empty dataframe but correct columns
    empty_df_with_cols = pd.DataFrame(columns=['Team', 'Status', 'ID'])
    summary_empty_df = calculate_team_status_summary(empty_df_with_cols)
    assert summary_empty_df.empty, "Test Case 3 Failed: Summary should be empty for an empty input DataFrame."
    assert list(summary_empty_df.columns) == expected_cols, \
        f"Test Case 3 Failed: Expected columns {expected_cols} for empty input, got {list(summary_empty_df.columns)}"
    print("Test Case 3 (Empty DataFrame with Columns) Passed.")

    print("All test_calculate_team_status_summary tests passed.")

def _get_week_display_str(year_week_str: str) -> str:
    """Helper to convert 'YYYY-Www' to 'YYYY-Www (Mon DD - Sun DD)'."""
    try:
        start_date = datetime.strptime(year_week_str + '-1', "%G-W%V-%u") # %u for Monday=1
        end_date = start_date + timedelta(days=6)
        return f"{year_week_str} ({start_date.strftime('%b %d')} - {end_date.strftime('%b %d, %Y')})"
    except ValueError:
        return year_week_str # Fallback if parsing fails (should not happen with correct Year-Week)


def calculate_srs_created_per_week(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the number of SRs created per week from a DataFrame.
    Now includes categorization by status and a week display string.

    Args:
        df: DataFrame containing SR data with a 'Created On' column.
            May optionally contain a 'Status' column.

    Returns:
        A DataFrame with columns ['Year-Week', 'WeekDisplay', 'StatusCategory' (optional), 'Number of SRs']
        Sorted appropriately. Returns an empty DataFrame if 'Created On'
        is missing or data cannot be processed.
    """
    if 'Created On' not in df.columns:
        # Determine expected columns for empty df based on original df's columns
        cols = ['Year-Week', 'WeekDisplay', 'Number of SRs']
        if 'Status' in df.columns: # If original df had Status, expect StatusCategory
            cols.insert(2, 'StatusCategory')
        return pd.DataFrame(columns=cols)

    processed_df = df.copy()
    processed_df['Created On'] = pd.to_datetime(processed_df['Created On'], errors='coerce')
    processed_df.dropna(subset=['Created On'], inplace=True)

    if processed_df.empty:
        cols = ['Year-Week', 'WeekDisplay', 'Number of SRs']
        if 'Status' in df.columns:
            cols.insert(2, 'StatusCategory')
        return pd.DataFrame(columns=cols)

    processed_df['Year-Week'] = processed_df['Created On'].dt.strftime('%G-W%V')

    group_by_cols = ['Year-Week']
    if 'Status' in processed_df.columns:
        processed_df['StatusCategory'] = np.select(
            [processed_df['Status'].fillna('').str.lower().isin(['closed', 'cancelled'])],
            ['Closed/Cancelled'],
            default='New/Pending'
        )
        group_by_cols.append('StatusCategory')

    srs_per_week = processed_df.groupby(group_by_cols).size().reset_index(name='Number of SRs')

    # Add WeekDisplay column
    if not srs_per_week.empty:
        srs_per_week['WeekDisplay'] = srs_per_week['Year-Week'].apply(_get_week_display_str)
    else: # Handle empty srs_per_week after grouping
        srs_per_week['WeekDisplay'] = pd.Series(dtype='str')


    # Sorting
    sort_cols = ['Year-Week']
    if 'StatusCategory' in srs_per_week.columns:
        sort_cols.append('StatusCategory')
    srs_per_week = srs_per_week.sort_values(by=sort_cols).reset_index(drop=True)

    # Reorder columns to make WeekDisplay appear after Year-Week
    if 'WeekDisplay' in srs_per_week.columns:
        all_cols = srs_per_week.columns.tolist()
        # Remove WeekDisplay and Year-Week to reinsert them at the beginning
        if 'Year-Week' in all_cols: all_cols.remove('Year-Week')
        if 'WeekDisplay' in all_cols: all_cols.remove('WeekDisplay')

        final_cols_order = ['Year-Week', 'WeekDisplay'] + all_cols
        # Ensure all original columns are present in final_cols_order to avoid KeyError
        # This can happen if srs_per_week was empty and some columns were not created.
        # A more robust way is to define the full expected order.

        expected_final_cols = ['Year-Week', 'WeekDisplay']
        if 'StatusCategory' in processed_df.columns: # Check original df for Status presence
             expected_final_cols.append('StatusCategory')
        expected_final_cols.append('Number of SRs')

        # Filter final_cols_order to only include columns that actually exist in srs_per_week
        srs_per_week = srs_per_week[[col for col in expected_final_cols if col in srs_per_week.columns]]

    return srs_per_week

def test_calculate_srs_created_per_week():
    """Tests for the calculate_srs_created_per_week function."""
    print("Running test_calculate_srs_created_per_week...")

    # 1. Basic functionality (No Status column)
    data1 = {'Created On': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-08', '2023-01-08'])}
    df1 = pd.DataFrame(data1)
    result1 = calculate_srs_created_per_week(df1)
    expected1_data = {
        'Year-Week': ['2022-W52', '2023-W01'],
        'WeekDisplay': [
            '2022-W52 (Dec 26 - Jan 01, 2023)', # Corrected format
            '2023-W01 (Jan 02 - Jan 08, 2023)'  # Corrected format
        ],
        'Number of SRs': [1, 3]
    }
    expected1 = pd.DataFrame(expected1_data)
    pd.testing.assert_frame_equal(result1, expected1)
    print("  Test Case 1 (Basic functionality - No Status) Passed.")

    # 2. Missing 'Created On' column
    df2 = pd.DataFrame({'SomeOtherColumn': [1, 2], 'Status': ['Open', 'Closed']})
    result2 = calculate_srs_created_per_week(df2)
    expected2 = pd.DataFrame(columns=['Year-Week', 'WeekDisplay', 'StatusCategory', 'Number of SRs'])
    pd.testing.assert_frame_equal(result2, expected2, check_dtype=False)
    print("  Test Case 2 (Missing 'Created On' column) Passed.")

    # 3. Dates that cannot be parsed (some valid, with Status)
    data3 = {'Created On': [
        pd.Timestamp('2023-01-01'),
        None,
        pd.Timestamp('2023-01-03'),
        'completely invalid date string',
        pd.Timestamp('2023-01-09 10:00:00')
    ], 'Status': ['Open', 'New', 'Closed', 'Pending', 'Cancelled']}
    df3 = pd.DataFrame(data3)
    result3 = calculate_srs_created_per_week(df3)
    expected3_data = {
        'Year-Week': ['2022-W52', '2023-W01', '2023-W02'],
        'WeekDisplay': [
            '2022-W52 (Dec 26 - Jan 01, 2023)',
            '2023-W01 (Jan 02 - Jan 08, 2023)',
            '2023-W02 (Jan 09 - Jan 15, 2023)'
        ],
        'StatusCategory': ['New/Pending', 'Closed/Cancelled', 'Closed/Cancelled'],
        'Number of SRs': [1, 1, 1]
    }
    expected3 = pd.DataFrame(expected3_data)
    pd.testing.assert_frame_equal(result3, expected3)
    print("  Test Case 3 (Some unparseable dates, with Status) Passed.")

    # 4. All dates invalid (with Status)
    data4 = {'Created On': ['not a date', None, ''], 'Status': ['Open', 'New', 'Closed']}
    df4 = pd.DataFrame(data4)
    result4 = calculate_srs_created_per_week(df4)
    expected4 = pd.DataFrame(columns=['Year-Week', 'WeekDisplay', 'StatusCategory', 'Number of SRs'])
    pd.testing.assert_frame_equal(result4, expected4, check_dtype=False)
    print("  Test Case 4 (All dates invalid, with Status) Passed.")

    # 5. Empty input DataFrame (with Status column defined)
    df5 = pd.DataFrame(columns=['Created On', 'Status'])
    result5 = calculate_srs_created_per_week(df5)
    expected5 = pd.DataFrame(columns=['Year-Week', 'WeekDisplay', 'StatusCategory', 'Number of SRs'])
    pd.testing.assert_frame_equal(result5, expected5, check_dtype=False)
    print("  Test Case 5 (Empty input DataFrame, with Status) Passed.")

    # 6. Correct sorting (with Status)
    data6 = {
        'Created On': pd.to_datetime(['2023-01-15', '2023-01-01', '2023-01-08', '2023-01-01']),
        'Status': ['Closed', 'Open', 'Pending', 'Cancelled']
    }
    df6 = pd.DataFrame(data6)
    result6 = calculate_srs_created_per_week(df6)
    expected6_data = {
        'Year-Week':    ['2022-W52',        '2022-W52',         '2023-W01',    '2023-W02'],
        'WeekDisplay': [
            '2022-W52 (Dec 26 - Jan 01, 2023)', '2022-W52 (Dec 26 - Jan 01, 2023)',
            '2023-W01 (Jan 02 - Jan 08, 2023)', '2023-W02 (Jan 09 - Jan 15, 2023)'
        ],
        'StatusCategory': ['Closed/Cancelled','New/Pending',    'New/Pending', 'Closed/Cancelled'],
        'Number of SRs': [1,                 1,                  1,             1]
    }
    # Sort expected the same way the function does
    expected6 = pd.DataFrame(expected6_data).sort_values(by=['Year-Week', 'StatusCategory']).reset_index(drop=True)
    pd.testing.assert_frame_equal(result6, expected6)
    print("  Test Case 6 (Correct sorting, with Status) Passed.")

    # 7. Different date/time formats and mixed status cases (including NaN status)
    data7 = {'Created On': [
        '2024-01-01T00:00:00',
        '2025-07-05T07:33:00',
        '2025-07-06T08:00:00',
        '2024-01-02T10:00:00',
        '2024-01-03T11:00:00'
    ], 'Status': ['new', 'CLOSED', 'CaNcElLeD', pd.NA, 'Active']}
    df7 = pd.DataFrame(data7)
    result7 = calculate_srs_created_per_week(df7)
    expected7_data = {
        'Year-Week':    ['2024-W01',    '2025-W27'],
        'WeekDisplay': [
            '2024-W01 (Jan 01 - Jan 07, 2024)',
            '2025-W27 (Jun 30 - Jul 06, 2025)' # July 5/6 2025 is W27
        ],
        'StatusCategory': ['New/Pending', 'Closed/Cancelled'],
        'Number of SRs': [3,             2]
    }
    expected7 = pd.DataFrame(expected7_data)
    pd.testing.assert_frame_equal(result7, expected7)
    print("  Test Case 7 (Mixed formats, statuses, NaN status) Passed.")

    # 8. Year boundary (ISO week, with Status)
    data8 = {
        'Created On': pd.to_datetime(['2023-12-31', '2024-01-01', '2024-12-29', '2024-12-30', '2025-01-01']),
        'Status': ['Open', 'Closed', 'Cancelled', 'Pending', 'New']
    }
    df8 = pd.DataFrame(data8)
    result8 = calculate_srs_created_per_week(df8)
    expected8_data = {
        'Year-Week':    ['2023-W52',    '2024-W01',         '2024-W52',         '2025-W01'],
        'WeekDisplay': [
            '2023-W52 (Dec 25 - Dec 31, 2023)', # Corrected
            '2024-W01 (Jan 01 - Jan 07, 2024)',
            '2024-W52 (Dec 23 - Dec 29, 2024)', # Corrected
            '2025-W01 (Dec 30 - Jan 05, 2025)'  # Corrected
        ],
        'StatusCategory': ['New/Pending', 'Closed/Cancelled', 'Closed/Cancelled', 'New/Pending'],
        'Number of SRs': [1,             1,                  1,                  2]
    }
    expected8 = pd.DataFrame(expected8_data)
    pd.testing.assert_frame_equal(result8, expected8)
    print("  Test Case 8 (Year boundary ISO week, with Status) Passed.")

    print("All test_calculate_srs_created_per_week tests passed.")
def calculate_srs_created_and_closed_per_week(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the number of SRs created and closed per week from a DataFrame.

    Args:
        df: DataFrame containing SR data with 'Created On', 'LastModDateTime', and 'Status' columns.

    Returns:
        A DataFrame with columns ['Year-Week', 'WeekDisplay', 'Count', 'Category']
        where 'Category' is 'Created' or 'Closed'.
        Sorted appropriately. Returns an empty DataFrame if essential columns
        are missing or data cannot be processed.
    """
    required_cols = ['Created On', 'LastModDateTime', 'Status']
    if not all(col in df.columns for col in required_cols):
        # Consider logging this issue if a logging mechanism is available
        print("Warning: calculate_srs_created_and_closed_per_week missing required columns.")
        return pd.DataFrame(columns=['Year-Week', 'WeekDisplay', 'Count', 'Category'])

    # --- SRs Created ---
    df_created = df.copy()
    initial_created_count = len(df_created)
    df_created['Created On'] = pd.to_datetime(df_created['Created On'], errors='coerce', dayfirst=True, infer_datetime_format=True)
    df_created.dropna(subset=['Created On'], inplace=True)
    parsed_created_count = len(df_created)
    if initial_created_count > 0 and parsed_created_count < initial_created_count * 0.8: # Example: if more than 20% failed
        print(f"Warning: Significant number of 'Created On' dates failed to parse ({initial_created_count - parsed_created_count} out of {initial_created_count}).")


    if df_created.empty:
        srs_created_weekly = pd.DataFrame(columns=['Year-Week', 'Count']).astype({'Year-Week': 'str', 'Count': pd.Int64Dtype()})
    else:
        df_created['Year-Week'] = df_created['Created On'].dt.strftime('%G-W%V')
        srs_created_weekly = df_created.groupby('Year-Week').size().reset_index(name='Count') # Count is int here
    
    srs_created_weekly['Category'] = 'Created'

    # --- SRs Closed ---
    df_closed = df.copy()
    # Normalize status column for comparison
    if 'Status' in df_closed.columns:
        df_closed['Status_normalized'] = df_closed['Status'].astype(str).str.lower().str.strip()
    else: # Should not happen if required_cols check passed, but as a safeguard
        df_closed['Status_normalized'] = pd.Series(dtype='str')

    closed_statuses_normalized = ["closed","completed", "cancelled", "approval rejected", "rejected by ps"]
    df_closed = df_closed[df_closed['Status_normalized'].isin(closed_statuses_normalized)]
    
    initial_closed_count = len(df_closed) # Count after filtering by normalized status
    df_closed['LastModDateTime'] = pd.to_datetime(df_closed['LastModDateTime'], errors='coerce', dayfirst=True, infer_datetime_format=True)
    df_closed.dropna(subset=['LastModDateTime'], inplace=True)
    parsed_closed_count = len(df_closed)
    if initial_closed_count > 0 and parsed_closed_count < initial_closed_count * 0.8: # Example: if more than 20% failed
        print(f"Warning: Significant number of 'LastModDateTime' dates failed to parse for closed SRs ({initial_closed_count - parsed_closed_count} out of {initial_closed_count}).")

    if df_closed.empty:
        srs_closed_weekly = pd.DataFrame(columns=['Year-Week', 'Count']).astype({'Year-Week': 'str', 'Count': pd.Int64Dtype()})
    else:
        df_closed['Year-Week'] = df_closed['LastModDateTime'].dt.strftime('%G-W%V')
        srs_closed_weekly = df_closed.groupby('Year-Week').size().reset_index(name='Count') # Count is int here
        
    srs_closed_weekly['Category'] = 'Closed'

    # Clean up temporary normalized status column if it exists
    if 'Status_normalized' in df_closed.columns:
        df_closed = df_closed.drop(columns=['Status_normalized'])


    # --- Combine and add WeekDisplay ---
    combined_df = pd.concat([srs_created_weekly, srs_closed_weekly], ignore_index=True)

    if combined_df.empty:
        return pd.DataFrame(columns=['Year-Week', 'WeekDisplay', 'Count', 'Category'])

    # Add WeekDisplay column
    # Ensure 'Year-Week' column exists before applying _get_week_display_str
    if 'Year-Week' in combined_df.columns:
        combined_df['WeekDisplay'] = combined_df['Year-Week'].apply(_get_week_display_str)
    else: # Should not happen if srs_created_weekly or srs_closed_weekly had data
        combined_df['WeekDisplay'] = pd.Series(dtype='str')


    # Sorting and final column order
    combined_df = combined_df.sort_values(by=['Year-Week', 'Category']).reset_index(drop=True)
    
    final_columns = ['Year-Week', 'WeekDisplay', 'Count', 'Category']
    # Filter to only include columns that actually exist, in the desired order
    combined_df = combined_df[[col for col in final_columns if col in combined_df.columns]]

    return combined_df


def test_calculate_srs_created_and_closed_per_week():
    """Tests for the calculate_srs_created_and_closed_per_week function."""
    print("Running test_calculate_srs_created_and_closed_per_week...")

    # Test Case 1: Basic scenario with created and closed SRs
    data1 = {
        'Created On': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-08', '2023-01-09']),
        'LastModDateTime': pd.to_datetime([None, '2023-01-03', '2023-01-10', '2023-01-10']),
        'Status': ['Open', 'Closed', 'Cancelled', 'Rejected by PS']
    }
    df1 = pd.DataFrame(data1)
    result1 = calculate_srs_created_and_closed_per_week(df1)
    expected1_data = {
        'Year-Week': ['2022-W52', '2023-W01', '2023-W01', '2023-W02', '2023-W02'],
        'WeekDisplay': [
            _get_week_display_str('2022-W52'),  # SR created on Jan 1st
            _get_week_display_str('2023-W01'),  # SR created on Jan 2nd
            _get_week_display_str('2023-W01'),  # SR closed on Jan 3rd
            _get_week_display_str('2023-W02'),  # SR created on Jan 8th
            _get_week_display_str('2023-W02'),  # SR created on Jan 9th
        ],
        'Count': [1, 1, 1, 2, 2], # Corrected: 1 created in W52, 1 created W01, 1 closed W01, 2 created W02, 2 closed W02
        'Category': ['Created', 'Created', 'Closed', 'Created', 'Closed']
    }
    # Rebuild expected1_data based on how the function aggregates
    # SR1: Created 2023-01-01 (Sun, 2022-W52)
    # SR2: Created 2023-01-02 (Mon, 2023-W01), Closed 2023-01-03 (Tue, 2023-W01)
    # SR3: Created 2023-01-08 (Sun, 2023-W01), Cancelled 2023-01-10 (Tue, 2023-W02)
    # SR4: Created 2023-01-09 (Mon, 2023-W02), Rejected 2023-01-10 (Tue, 2023-W02)

    # Created Counts:
    #   2022-W52: 1 (SR1)
    #   2023-W01: 2 (SR2, SR3)
    #   2023-W02: 1 (SR4)
    # Closed Counts:
    #   2023-W01: 1 (SR2)
    #   2023-W02: 2 (SR3, SR4)
    
    expected1_df_data = [
        {'Year-Week': '2022-W52', 'WeekDisplay': _get_week_display_str('2022-W52'), 'Count': 1, 'Category': 'Created'},
        {'Year-Week': '2023-W01', 'WeekDisplay': _get_week_display_str('2023-W01'), 'Count': 2, 'Category': 'Created'}, # Corrected
        {'Year-Week': '2023-W01', 'WeekDisplay': _get_week_display_str('2023-W01'), 'Count': 1, 'Category': 'Closed'},
        {'Year-Week': '2023-W02', 'WeekDisplay': _get_week_display_str('2023-W02'), 'Count': 1, 'Category': 'Created'}, # Corrected
        {'Year-Week': '2023-W02', 'WeekDisplay': _get_week_display_str('2023-W02'), 'Count': 2, 'Category': 'Closed'},
    ]
    expected1 = pd.DataFrame(expected1_df_data)
    # Sort expected the same way the function does: by Year-Week, then Category
    expected1 = expected1.sort_values(by=['Year-Week', 'Category']).reset_index(drop=True)

    # Debug: print both dataframes if they don't match
    # if not result1.equals(expected1):
    #     print("--- RESULT 1 (Actual) ---")
    #     print(result1)
    #     print("--- EXPECTED 1 ---")
    #     print(expected1)

    pd.testing.assert_frame_equal(result1, expected1, check_like=True) # check_like ignores order of rows if columns match
    print("  Test Case 1 (Basic scenario) Passed.")

    # Test Case 2: Missing required columns
    df2 = pd.DataFrame({'Created On': [pd.to_datetime('2023-01-01')]})
    result2 = calculate_srs_created_and_closed_per_week(df2)
    expected2 = pd.DataFrame(columns=['Year-Week', 'WeekDisplay', 'Count', 'Category'])
    pd.testing.assert_frame_equal(result2, expected2, check_dtype=False)
    print("  Test Case 2 (Missing required columns) Passed.")

    # Test Case 3: No data results in empty dataframe
    df3 = pd.DataFrame(columns=['Created On', 'LastModDateTime', 'Status'])
    result3 = calculate_srs_created_and_closed_per_week(df3)
    expected3 = pd.DataFrame(columns=['Year-Week', 'WeekDisplay', 'Count', 'Category'])
    pd.testing.assert_frame_equal(result3, expected3, check_dtype=False)
    print("  Test Case 3 (No data) Passed.")

    # Test Case 4: Only created SRs, no closed SRs
    data4 = {
        'Created On': pd.to_datetime(['2023-03-01', '2023-03-02']),
        'LastModDateTime': [None, None],
        'Status': ['Open', 'Pending']
    }
    df4 = pd.DataFrame(data4)
    result4 = calculate_srs_created_and_closed_per_week(df4)
    expected4_data = {
        'Year-Week': ['2023-W09'],
        'WeekDisplay': [_get_week_display_str('2023-W09')],
        'Count': [2],
        'Category': ['Created']
    }
    expected4 = pd.DataFrame(expected4_data).astype({'Count': pd.Int64Dtype()})
    pd.testing.assert_frame_equal(result4, expected4, check_like=True)
    print("  Test Case 4 (Only created SRs) Passed.")

    # Test Case 5: Only closed SRs, no created SRs (e.g., all creation dates are invalid)
    data5 = {
        'Created On': [None, 'invalid_date'],
        'LastModDateTime': pd.to_datetime(['2023-03-05', '2023-03-06']),
        'Status': ['Closed', 'Cancelled']
    }
    df5 = pd.DataFrame(data5)
    result5 = calculate_srs_created_and_closed_per_week(df5)
    # Corrected expected data for Test Case 5:
    # '2023-03-05' is 2023-W09, '2023-03-06' is 2023-W10
    expected5_df_data = [
        {'Year-Week': '2023-W09', 'WeekDisplay': _get_week_display_str('2023-W09'), 'Count': 1, 'Category': 'Closed'},
        {'Year-Week': '2023-W10', 'WeekDisplay': _get_week_display_str('2023-W10'), 'Count': 1, 'Category': 'Closed'},
    ]
    expected5 = pd.DataFrame(expected5_df_data).astype({'Count': pd.Int64Dtype()})
    # Ensure sorting matches function output if the order of definition isn't naturally sorted
    expected5 = expected5.sort_values(by=['Year-Week', 'Category']).reset_index(drop=True)
    pd.testing.assert_frame_equal(result5, expected5, check_like=True)
    print("  Test Case 5 (Only closed SRs) Passed.")
    
    # Test Case 6: SRs closed with other statuses (should not be counted as 'Closed')
    data6 = {
        'Created On': pd.to_datetime(['2023-04-01']),
        'LastModDateTime': pd.to_datetime(['2023-04-03']),
        'Status': ['Pending Resolution'] # Not one of the specified closed statuses
    }
    df6 = pd.DataFrame(data6)
    result6 = calculate_srs_created_and_closed_per_week(df6)
    expected6_data = { # Only the created SR should appear
        'Year-Week': ['2023-W13'],
        'WeekDisplay': [_get_week_display_str('2023-W13')], # April 1st is in W13
        'Count': [1],
        'Category': ['Created']
    }
    expected6 = pd.DataFrame(expected6_data).astype({'Count': pd.Int64Dtype()})
    pd.testing.assert_frame_equal(result6, expected6, check_like=True)
    print("  Test Case 6 (SRs with non-closed statuses) Passed.")

    # Test Case 7: Mixed valid and invalid dates for Created On and LastModDateTime
    data7 = {
        'Created On': pd.to_datetime(['2023-05-01', 'invalid', '2023-05-08'], errors='coerce'),
        'LastModDateTime': pd.to_datetime(['invalid_date', '2023-05-03', '2023-05-10'], errors='coerce'),
        'Status': ['Closed', 'Closed', 'Cancelled'] # SR with 'invalid' Created On still processed for closure if LastModDateTime is valid
    }
    df7 = pd.DataFrame(data7)
    result7 = calculate_srs_created_and_closed_per_week(df7)

    # Breakdown for data7:
    # SR1: Created 2023-05-01 (W18), LastModDateTime NaT, Status Closed. -> Created W18 (1)
    # SR2: Created NaT, LastModDateTime 2023-05-03 (W18), Status Closed. -> Closed W18 (1)
    # SR3: Created 2023-05-08 (W19), LastModDateTime 2023-05-10 (W19), Status Cancelled. -> Created W19 (1), Closed W19 (1)

    expected7_df_data = [
        {'Year-Week': '2023-W18', 'WeekDisplay': _get_week_display_str('2023-W18'), 'Count': 1, 'Category': 'Created'}, # SR1
        {'Year-Week': '2023-W18', 'WeekDisplay': _get_week_display_str('2023-W18'), 'Count': 1, 'Category': 'Closed'},  # SR2
        {'Year-Week': '2023-W19', 'WeekDisplay': _get_week_display_str('2023-W19'), 'Count': 1, 'Category': 'Created'}, # SR3
        {'Year-Week': '2023-W19', 'WeekDisplay': _get_week_display_str('2023-W19'), 'Count': 1, 'Category': 'Closed'},  # SR3
    ]
    # In this case, both created and closed have data, so Count should be int64, not Int64.
    expected7 = pd.DataFrame(expected7_df_data)
    expected7 = expected7.sort_values(by=['Year-Week', 'Category']).reset_index(drop=True)
    pd.testing.assert_frame_equal(result7, expected7, check_like=True)
    print("  Test Case 7 (Mixed valid/invalid dates) Passed.")

    # Test Case 8: Varied status casing and whitespace
    data8 = {
        'Created On': pd.to_datetime(['2023-06-01', '2023-06-02', '2023-06-03', '2023-06-04']),
        'LastModDateTime': pd.to_datetime(['2023-06-05', '2023-06-06', '2023-06-07', '2023-06-08']),
        'Status': ['Closed  ', '  cancelled', 'APPROVAL REJECTED', 'rejected by ps','Completed']
    }
    df8 = pd.DataFrame(data8)
    result8 = calculate_srs_created_and_closed_per_week(df8)
    expected8_data = [
        {'Year-Week': '2023-W22', 'WeekDisplay': _get_week_display_str('2023-W22'), 'Count': 4, 'Category': 'Created'},
        {'Year-Week': '2023-W23', 'WeekDisplay': _get_week_display_str('2023-W23'), 'Count': 4, 'Category': 'Closed'},
    ]
    expected8 = pd.DataFrame(expected8_data)
    expected8 = expected8.sort_values(by=['Year-Week', 'Category']).reset_index(drop=True)
    pd.testing.assert_frame_equal(result8, expected8, check_like=True)
    print("  Test Case 8 (Varied status casing and whitespace) Passed.")

    # Test Case 9: Valid LastModDateTime but non-closing status
    data9 = {
        'Created On': pd.to_datetime(['2023-07-01']),
        'LastModDateTime': pd.to_datetime(['2023-07-03']),
        'Status': ['Pending Investigation']
    }
    df9 = pd.DataFrame(data9)
    result9 = calculate_srs_created_and_closed_per_week(df9)
    expected9_data = [
        {'Year-Week': '2023-W26', 'WeekDisplay': _get_week_display_str('2023-W26'), 'Count': 1, 'Category': 'Created'},
    ]
    expected9 = pd.DataFrame(expected9_data).astype({'Count': pd.Int64Dtype()})
    pd.testing.assert_frame_equal(result9, expected9, check_like=True)
    print("  Test Case 9 (Valid LastModDateTime, non-closing status) Passed.")

    # Test Case 10: Closing status but invalid/missing LastModDateTime
    data10 = {
        'Created On': pd.to_datetime(['2023-08-01', '2023-08-02']),
        'LastModDateTime': [None, 'invalid_date'],
        'Status': ['Closed', 'Cancelled']
    }
    df10 = pd.DataFrame(data10)
    result10 = calculate_srs_created_and_closed_per_week(df10)
    expected10_data = [
        {'Year-Week': '2023-W31', 'WeekDisplay': _get_week_display_str('2023-W31'), 'Count': 2, 'Category': 'Created'},
    ]
    expected10 = pd.DataFrame(expected10_data).astype({'Count': pd.Int64Dtype()})
    pd.testing.assert_frame_equal(result10, expected10, check_like=True)
    print("  Test Case 10 (Closing status, invalid LastModDateTime) Passed.")

    # Test Case 11: Ambiguous date format DD/MM/YYYY vs MM/DD/YYYY (testing dayfirst=True)
    # 01/02/2023 should be Feb 1, 2023. 13/01/2023 should be Jan 13, 2023
    data11 = {
        'Created On': ['01/02/2023', '13/01/2023'], # Feb 1 (W05), Jan 13 (W02)
        'LastModDateTime': ['02/02/2023', '14/01/2023'], # Feb 2 (W05), Jan 14 (W02)
        'Status': ['Closed', 'Cancelled']
    }
    df11 = pd.DataFrame(data11)
    # Convert to datetime explicitly here for the test setup if needed,
    # or rely on the function's internal parsing with dayfirst=True.
    # The function itself will handle the parsing.
    result11 = calculate_srs_created_and_closed_per_week(df11)
    expected11_data = [
        {'Year-Week': '2023-W02', 'WeekDisplay': _get_week_display_str('2023-W02'), 'Count': 1, 'Category': 'Created'}, # Jan 13
        {'Year-Week': '2023-W02', 'WeekDisplay': _get_week_display_str('2023-W02'), 'Count': 1, 'Category': 'Closed'},  # Jan 14
        {'Year-Week': '2023-W05', 'WeekDisplay': _get_week_display_str('2023-W05'), 'Count': 1, 'Category': 'Created'}, # Feb 1
        {'Year-Week': '2023-W05', 'WeekDisplay': _get_week_display_str('2023-W05'), 'Count': 1, 'Category': 'Closed'},  # Feb 2
    ]
    expected11 = pd.DataFrame(expected11_data).sort_values(by=['Year-Week', 'Category']).reset_index(drop=True)
    pd.testing.assert_frame_equal(result11, expected11, check_like=True)
    print("  Test Case 11 (Ambiguous dates with dayfirst=True) Passed.")


    print("All test_calculate_srs_created_and_closed_per_week tests passed.")

if __name__ == '__main__':
    test_calculate_team_status_summary()
    test_case_count_calculation_and_filtering()
    test_calculate_srs_created_per_week()
    test_calculate_srs_created_and_closed_per_week()  # Corrected this line if it was the source of a typo
    print("All utils.py tests passed successfully when run directly.")
