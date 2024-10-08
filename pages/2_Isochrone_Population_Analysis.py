## -------------------------------------------------------------------------------------------------------------------------------------
## Import packages
## -------------------------------------------------------------------------------------------------------------------------------------

# Import packages
import os
import pandas as pd
import geopandas as gpd
import time
from tqdm import tqdm
from datetime import datetime
import re
import streamlit as st

# ---------------------------------------------------------------------------
# Introduce page
# ---------------------------------------------------------------------------

# Title and Description
st.title("Isochrone Population Analysis")
st.markdown("""
This application calculates the population, segmented by age group, for each isochrone generated in the previous step.

Currently, the application only works with **CBS PC4 data** from 2022 and 2023. Including PC6 data is being worked on.
""")

# ---------------------------------------------------------------------------
# Define functions
# ---------------------------------------------------------------------------

def generate_and_validate_filename(filename):
    base_name = os.path.basename(filename)
    
    valid_transport_methods = ['driving', 'cycling', 'walking', 'public_transport']
    pattern = r'({})_(\d+)_minutes_output'.format('|'.join(valid_transport_methods))
    
    match = re.search(pattern, base_name)
    
    if not match:
        raise ValueError("Filename does not match the expected format: <transport_method>_<number_of_minutes>_minutes_output.gpkg")
    
    transport_method = match.group(1)
    duration = f"{match.group(2)}_minutes"
    
    new_filename = f"{transport_method}_{duration}"
    return new_filename

def clear_session_state():
    st.session_state.geo_dfs = {}
    st.session_state.gdf_names_dict = {}
    st.session_state.cbs_geo_df = None
    st.session_state.year = ""

# ---------------------------------------------------------------------------
# Read files
# ---------------------------------------------------------------------------

# Initialize session state variables
if 'geo_dfs' not in st.session_state:
    st.session_state.geo_dfs = {}
if 'gdf_names_dict' not in st.session_state:
    st.session_state.gdf_names_dict = {}
if 'cbs_geo_df' not in st.session_state:
    st.session_state.cbs_geo_df = None
if 'year' not in st.session_state:
    st.session_state.year = ""

# Add a session state variable for data read status
if 'data_read' not in st.session_state:
    st.session_state.data_read = False

# File uploader prompt for general GeoDataFrames
uploaded_files = st.file_uploader(
    "Upload the GPKG file(s) generated in step 1 here:",
    type=["gpkg"],
    accept_multiple_files=True
)

geo_dfs = {}
gdf_names_dict = {}

if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            new_file_name = generate_and_validate_filename(uploaded_file.name)
            gdf = gpd.read_file(uploaded_file)

            if 'geometry' in gdf.columns:
                gdf = gdf.rename(columns={'geometry': new_file_name})
                gdf = gdf.set_geometry(new_file_name)

            if new_file_name not in gdf.columns or not gpd.GeoSeries(gdf[new_file_name]).is_valid.all():
                raise ValueError(f"The geometry column '{new_file_name}' is not valid in the file '{uploaded_file.name}'.")

            gdf.to_crs('EPSG:28992', inplace=True)
                
            geo_dfs[new_file_name] = gdf
            gdf_names_dict[new_file_name] = uploaded_file.name
            
        except Exception as e:
            st.error(f'Error processing file "{uploaded_file.name}": {e}')

    st.success("Files imported successfully")

# Prompt for CBS PC4 or PC6 file
cbs_file = st.file_uploader(
    "Upload the CBS PC4 GPKG file here",
    type=["gpkg"]
)

# Input for year using a text field to start empty
current_year = datetime.now().year
st.session_state.year = st.text_input(
    f"Enter the year the document is from (between 2000 and {current_year})", 
    value=st.session_state.get('year', '')
)

# Display the "Read data" button at the start
if st.button("Read data"):
    # Check if both cbs_file and year are provided
    if cbs_file and st.session_state.year:
        read_info = st.empty()
        read_info.info("Reading data...")

        try:
            year_as_int = int(st.session_state.year)
            if not 2000 <= year_as_int <= current_year:
                raise ValueError(f"Year must be between 2000 and {current_year}")

            cbs_geo_df = gpd.read_file(cbs_file)
            cbs_geo_df['year'] = year_as_int
            

            read_info.empty()

            st.success("Data read successfully.")
            
            # Additional processing...
            cbs_geo_df = cbs_geo_df.fillna(0)

            for col in cbs_geo_df.select_dtypes(include=['number']).columns:
                cbs_geo_df[col] = cbs_geo_df[col].apply(lambda x: max(x, 0))

            if 'postcode6' in cbs_geo_df.columns:
                cbs_geo_df = cbs_geo_df.rename(columns={'postcode6': 'postcode'})

            try:
                cbs_geo_df.to_crs('EPSG:28992', inplace=True)
                
            except Exception as e:
                st.error(f'Error converting CBS GeoDataFrame to CRS 28992: {e}')

            
            # Set data_read to True once everything is processed successfully
            st.session_state.data_read = True
            st.session_state.geo_dfs = geo_dfs
            st.session_state.gdf_names_dict = gdf_names_dict
            st.session_state.cbs_geo_df = cbs_geo_df

            # Fix: Reset data_downloaded here
            st.session_state.data_downloaded = False

        except Exception as e:
            read_info.empty()
            st.error(f"An error occurred while reading the CBS data: {e}")
            st.session_state.data_read = False  # Ensure it's False if an error occurs
    else:
        st.warning("Please upload the CBS file and enter the year before proceeding.")



# ---------------------------------------------------------------------------
# Calculate
# ---------------------------------------------------------------------------

# Check if data has been successfully read in the previous step
if 'data_read' not in st.session_state:
    st.session_state.data_read = False

# Initialize a flag to check if data has been downloaded
if 'data_downloaded' not in st.session_state:
    st.session_state.data_downloaded = False

# Display a button to calculate population per isochrone only if data has been read and not yet downloaded
if st.session_state.data_read and not st.session_state.data_downloaded and st.button("Calculate population per isochrone"):

    cbs_geo_df = st.session_state.cbs_geo_df  # Retrieve from session state
    
    # Display a "Calculating..." message
    calc_info = st.empty()
    calc_info.info("Calculating...")

    start_time = time.time()

    # Initialize new columns in cbs_geo_df for overlap area and percentage
    cbs_geo_df['overlap_area'] = 0.0
    cbs_geo_df['overlap_percentage'] = 0.0

    # Define age groups with corresponding labels
    age_groups = {
        'aantal_inwoners_0_tot_15_jaar': '0_15',
        'aantal_inwoners_15_tot_25_jaar': '15_25',
        'aantal_inwoners_25_tot_45_jaar': '25_45',
        'aantal_inwoners_45_tot_65_jaar': '45_65',
        'aantal_inwoners_65_jaar_en_ouder': '65_and_older',
        'aantal_inwoners': 'total_population'
    }

    # Initialize new columns for population calculation
    for age_group in age_groups.keys():
        cbs_geo_df[f'num_pop_calculated_{age_group}'] = 0.0

    # Create a dictionary to store transport intervals
    transport_intervals = {}
    for key in gdf_names_dict.keys():
        numbers = re.findall(r'\d+', key)
        if not numbers:
            print(f"Error: No numeric part found in {key}")
            continue
        
        minutes_str = numbers[0]
        first_numeric_idx = key.index(minutes_str)
        transport_type = key[:first_numeric_idx].rstrip('_')
        
        try:
            minutes = int(minutes_str)
        except ValueError:
            print(f"Unable to convert {minutes_str} to an integer")
            continue
        
        if transport_type not in transport_intervals:
            transport_intervals[transport_type] = []
        
        transport_intervals[transport_type].append(minutes)

    # Sort transport intervals and ensure they start with a zero interval
    for transport in transport_intervals:
        transport_intervals[transport].sort()
        if transport_intervals[transport][0] != 0:
            transport_intervals[transport] = [0] + transport_intervals[transport]
        transport_intervals[transport] = [
            (transport_intervals[transport][i], transport_intervals[transport][i + 1])
            for i in range(len(transport_intervals[transport]) - 1)
        ]

    all_output_dfs = []

    # Check for invalid geometries in cbs_geo_df
    invalid_geometries = cbs_geo_df[~cbs_geo_df.is_valid]
    if not invalid_geometries.empty:
        print("Invalid geometries found:")
        print(invalid_geometries)
        cbs_geo_df['geometry'] = cbs_geo_df['geometry'].apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)

    # Loop through each GeoDataFrame in geo_dfs
    for key, gdf in geo_dfs.items():
        minutes = int(re.findall(r'\d+', key)[0])
        first_numeric_idx = key.index(str(minutes))
        transport_type = key[:first_numeric_idx].rstrip('_')

        original_geometry = gdf.geometry
        total_iterations = len(gdf)

        # Use masking and vectorized operations for calculations
        with tqdm(total=total_iterations, desc=f"Processing {key}", unit="iteration") as pbar:
            for idx_df, buffer_row in gdf.iterrows():
                temp_gdf = gpd.GeoDataFrame(geometry=[buffer_row[f'{transport_type}_{minutes}_minutes']], crs='EPSG:28992')
            
                overlaps = cbs_geo_df['geometry'].intersection(temp_gdf.geometry.iloc[0])
                overlap_areas = overlaps.area
                areas = cbs_geo_df['geometry'].area
                percentages = overlap_areas / areas.replace(0, pd.NA)
                
                valid_overlap_mask = percentages > 0.0
                filtered_df = cbs_geo_df.loc[valid_overlap_mask].copy()
                filtered_df['overlap_percentage'] = percentages[valid_overlap_mask]

                # Calculate population numbers per age group based on overlap percentage
                for age_group, new_label in age_groups.items():
                    filtered_df[f'num_pop_calculated_{age_group}'] = filtered_df[age_group] * filtered_df['overlap_percentage']
                    total_population_age_group = filtered_df[f'num_pop_calculated_{age_group}'].sum()
                    output_column_name = f'{transport_type}_0_{minutes}m_num_pop_calculated_{new_label}'
                    gdf.at[idx_df, output_column_name] = total_population_age_group
                
                pbar.update(1)

        gdf.geometry = original_geometry
        gdf['year'] = st.session_state.year  # Ensure year is included in each dataframe
        output_columns = ['uid', 'year'] + [f'{transport_type}_0_{minutes}m_num_pop_calculated_{new_label}' for new_label in age_groups.values()]
        all_output_dfs.append(gdf[output_columns])

    # Combine all output DataFrames into a single final output DataFrame
    final_output_df = all_output_dfs[0]
    for df in all_output_dfs[1:]:
        final_output_df = final_output_df.merge(df, on=['uid', 'year'], how='left')

    age_groups_list = list(age_groups.values())
    final_output_dfs = {}

    # Process final output DataFrame for each transport type and interval
    for transport_type, intervals in transport_intervals.items():
        for i, (start_minute, end_minute) in enumerate(intervals):
            for age_group in age_groups_list:
                new_col_name = f'{transport_type}_{start_minute}_{end_minute}m_num_pop_calculated_{age_group}'
                if start_minute == 0:
                    if new_col_name not in final_output_df.columns:
                        final_output_df[new_col_name] = final_output_df.get(new_col_name, pd.Series([0]*len(final_output_df)))
                else:
                    prev_end_minute = intervals[i-1][1]
                    prev_col_name = f'{transport_type}_{0}_{prev_end_minute}m_num_pop_calculated_{age_group}'
                    current_col_name = f'{transport_type}_{0}_{end_minute}m_num_pop_calculated_{age_group}'
                    if new_col_name not in final_output_df.columns:
                        final_output_df[new_col_name] = (
                            final_output_df.get(current_col_name, pd.Series([0]*len(final_output_df))) -
                            final_output_df.get(prev_col_name, pd.Series([0]*len(final_output_df)))
                        )
                    check_result = final_output_df[new_col_name].equals(
                        final_output_df[current_col_name] - final_output_df[prev_col_name]
                    )
                    assert check_result, (
                        f"Check failed for {new_col_name}: "
                        f"{final_output_df[current_col_name].values[0]} - {final_output_df[prev_col_name].values[0]} != {final_output_df[new_col_name].values[0]}"
                    )
        final_output_dfs[transport_type] = final_output_df.copy()

    # Ensure 'uid' is the first column and 'year' is the second column
    cols = final_output_df.columns.tolist()
    if 'uid' in cols and 'year' in cols:
        cols.insert(1, cols.pop(cols.index('year')))  # Make 'year' the second column
        final_output_df = final_output_df[cols]

    # Convert DataFrame to CSV
    csv = final_output_df.to_csv(index=False)

    calc_info.empty()  # Clear the "Calculating..." message

    # Show the final output
    st.write("Final combined output:")
    st.write(final_output_df)

    # Clear progress info message and progress bar
    st.success(f"Population per isochrone calculated successfully in {time.time() - start_time:.2f} seconds.")

    def clear_session_state():
        st.session_state.data_downloaded = True  # Set the flag when the download button is clicked

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='final_output.csv',
        mime='text/csv',
        on_click=clear_session_state
    )

