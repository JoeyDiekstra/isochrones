## -------------------------------------------------------------------------------------------------------------------------------------
## Import packages
## -------------------------------------------------------------------------------------------------------------------------------------

# Import packages
import subprocess
import sys

# def install(package):
#     """Installs the package using conda."""
#     subprocess.check_call([sys.executable, "-m", "conda", "install", "--yes", package])

# # List of packages to ensure they are installed
# required_packages = [
#     "os", "pandas", "shapely", "geopandas", 
#     "matplotlib", "time", "tqdm", "datetime", "re"
# ]

# # Ensure all required packages are installed
# for package in required_packages:
#     try:
#         __import__(package)
#     except ImportError:
#         install(package)

# Now import the required modules
import os
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from datetime import datetime
import re
import streamlit as st

# -----------------------------------------------------------------------------
# Read the Geographic Data File
# -----------------------------------------------------------------------------

# Title and Description
st.title("Isochrone Analysis")
st.markdown("""
This application calculates the population, segmented by age group, for each isochrone generated in the previous step.

Choose between:
- **PC4**: Faster but less accurate.
- **PC6**: Slower but more precise.
""")


# -----------------------------------------------------------------------------
# Define functions
# -----------------------------------------------------------------------------

import streamlit as st
import os
import geopandas as gpd
import time
import re
import matplotlib.pyplot as plt

# # Function to plot CBS GeoDataFrame
# def plot_cbs_data(cbs_geo_df):
#     fig, ax = plt.subplots(figsize=(10, 10))
#     cbs_geo_df.plot(ax=ax)
#     ax.set_title("Plot of CBS Geo DataFrame")
#     st.pyplot(fig)

# Function to validate the filename according to specific rules
def generate_and_validate_filename(filename):
    base_name = os.path.basename(filename)
    
    # Valid transport methods
    valid_transport_methods = ['driving', 'cycling', 'walking', 'public_transport']
    
    # Regex pattern to find transport method and number of minutes
    pattern = r'({})_(\d+)_minutes_output'.format('|'.join(valid_transport_methods))
    
    match = re.search(pattern, base_name)
    
    if not match:
        raise ValueError("Filename does not match the expected format: <transport_method>_<number_of_minutes>_minutes_output.gpkg")
    
    transport_method = match.group(1)
    duration = f"{match.group(2)}_minutes"
    
    new_filename = f"{transport_method}_{duration}"
    return new_filename

# File uploader prompt for general GeoDataFrames
uploaded_files = st.file_uploader(
    "Upload the GPKG file(s) here",
    type=["gpkg"],
    accept_multiple_files=True
)

geo_dfs = {}
cbs_geo_df = None
gdf_names_dict = {}

if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            new_file_name = generate_and_validate_filename(uploaded_file.name)

            gdf = gpd.read_file(uploaded_file)

            # Check if a 'geometry' column exists, and rename it to the new filename
            if 'geometry' in gdf.columns:
                gdf = gdf.rename(columns={'geometry': new_file_name})
                gdf = gdf.set_geometry(new_file_name)

            # Ensure the renamed geometry column is valid
            if new_file_name not in gdf.columns or not gpd.GeoSeries(gdf[new_file_name]).is_valid.all():
                raise ValueError(f"The geometry column '{new_file_name}' is not valid in the file '{uploaded_file.name}'.")
                
            geo_dfs[new_file_name] = gdf
            gdf_names_dict[new_file_name] = uploaded_file.name  # Save the mappings
            
        except Exception as e:
            st.error(f'Error processing file "{uploaded_file.name}": {e}')

    st.success("Files imported successfully")

# Set CRS to 'EPSG:28992' for all GeoDataFrames in the dictionary using a single loop
for gdf_name, gdf in geo_dfs.items():
    st.write(f'{gdf_name}: {gdf.crs}')

# Prompt for CBS PC4 or PC6 file
cbs_file = st.file_uploader(
    "Upload the CBS PC4 or PC6 GPKG file",
    type=["gpkg"]
)

if cbs_file:
    start_time = time.time()
    
    progress_bar = st.progress(0)
    
    try:
        for percent_complete in range(100):
            time.sleep(0.01)  # Simulate file reading with a short delay.
            progress_bar.progress(percent_complete + 1)
            
        cbs_geo_df = gpd.read_file(cbs_file)
        
        st.success(f"CBS file '{cbs_file.name}' successfully loaded.")
        st.write(f"Time taken to read the GPKG file: {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        st.error(f'Error reading CBS file "{cbs_file.name}": {e}')
    finally:
        progress_bar.empty()  # Clear the progress bar once done.

if uploaded_files and cbs_file:
    # Set N/As to 0
    cbs_geo_df = cbs_geo_df.fillna(0)

    # Replace negative numbers with zero in numerical columns
    for col in cbs_geo_df.select_dtypes(include=['number']).columns:
        cbs_geo_df[col] = cbs_geo_df[col].apply(lambda x: max(x, 0))

    # Check if 'postcode6' column exists and rename it to 'postcode'
    if 'postcode6' in cbs_geo_df.columns:
        cbs_geo_df = cbs_geo_df.rename(columns={'postcode6': 'postcode'})

    try:
        # Set CRS to 'EPSG:28992'
        cbs_geo_df.to_crs('EPSG:28992', inplace=True)

        # Set CRS to 'EPSG:28992' for all GeoDataFrames in the dictionary using a single loop
        for gdf_name, gdf in geo_dfs.items():
            gdf.to_crs('EPSG:28992', inplace=True)
        
        st.success("All GeoDataFrames converted to CRS 28992 successfully.")

        # # Plot the CBS GeoDataFrame once conversion is successful
        # plot_cbs_data(cbs_geo_df)

    except Exception as e:
        st.error(f'Error converting GeoDataFrames to CRS 28992: {e}')

# ## -------------------------------------------------------------------------------------------------------------------------------------
# ## Plot data (no requirement; just for checking data)
# ## -------------------------------------------------------------------------------------------------------------------------------------

# # Plot data for each time interval
# for name, gdf in geo_dfs.items():
#     fig, ax = plt.subplots(figsize=(10, 10))
#     gdf.plot(ax=ax)
#     ax.set_title(f"Plot of {name}")
#     plt.show()


# ## -------------------------------------------------------------------------------------------------------------------------------------
# ## Transform data
# ## -------------------------------------------------------------------------------------------------------------------------------------

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

# Initialize new columns in cbs_geo_df for calculated population numbers per age group
for age_group in age_groups.keys():
    cbs_geo_df[f'num_pop_calculated_{age_group}'] = 0.0

# Create a dictionary to store transport intervals
transport_intervals = {}
for key in gdf_names_dict.keys():
    # Extract numeric parts from the key
    numbers = re.findall(r'\d+', key)
    
    if not numbers:
        print(f"Error: No numeric part found in {key}")
        continue
    
    minutes_str = numbers[0]
    first_numeric_idx = key.index(minutes_str)
    transport_type = key[:first_numeric_idx].rstrip('_')
    
    try:
        # Convert the numeric part to integer (minutes)
        minutes = int(minutes_str)
    except ValueError:
        print(f"Unable to convert {minutes_str} to an integer")
        continue
    
    # Add transport type and interval to the dictionary
    if transport_type not in transport_intervals:
        transport_intervals[transport_type] = []
    
    transport_intervals[transport_type].append(minutes)


# Sort transport intervals and ensure they start with a zero interval
for transport in transport_intervals:
    transport_intervals[transport].sort()
    if transport_intervals[transport][0] != 0:
        transport_intervals[transport] = [0] + transport_intervals[transport]
    # Create interval tuples for each transport type
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
    
    # Attempt to fix them
    cbs_geo_df['geometry'] = cbs_geo_df['geometry'].apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)



# Loop through each GeoDataFrame in geo_dfs
for key, gdf in geo_dfs.items():
    minutes = int(re.findall(r'\d+', key)[0])
    first_numeric_idx = key.index(str(minutes))
    transport_type = key[:first_numeric_idx].rstrip('_')

    original_geometry = gdf.geometry
    start_time = time.time()
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

    end_time = time.time()
    duration = end_time - start_time
    print(f"{key} processing time: {duration:.2f} seconds")
    gdf.geometry = original_geometry
    output_columns = ['uid'] + [f'{transport_type}_0_{minutes}m_num_pop_calculated_{new_label}' for new_label in age_groups.values()]
    all_output_dfs.append(gdf[output_columns])



# Combine all output DataFrames into a single final output DataFrame
final_output_df = all_output_dfs[0]
for df in all_output_dfs[1:]:
    final_output_df = final_output_df.merge(df, on='uid', how='left')

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

# Print final combined output DataFrame
print("Final combined output:")
print(final_output_df)



# Convert DataFrame to CSV
csv = final_output_df.to_csv(index=False)

# Create the download button
st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='final_output.csv',
    mime='text/csv',
)

# Print final combined output DataFrame
st.write("Final combined output:")
st.write(final_output_df)





# ## -------------------------------------------------------------------------------------------------------------------------------------
# ## Write data
# ## -------------------------------------------------------------------------------------------------------------------------------------

# # Define the output directory and filename with a timestamp
# output_dir = r"C:\Users\Joey.Diekstra\OneDrive - OC&C Strategy Consultants\Personal\python\location_analytics\output"
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# filename = f"{timestamp}_population_analysis.xlsx"

# # Create the complete file path
# file_path = os.path.join(output_dir, filename)

# # Write data to the specified Excel file
# final_output_df.to_excel(file_path, index=False)


## -------------------------------------------------------------------------------------------------------------------------------------
## Plotting data for visualisation purposes
## -------------------------------------------------------------------------------------------------------------------------------------


# # Extracting one GeoDataFrame from geo_dfs
# _, test_gdf = next(iter(geo_dfs.items()))

# # Ensuring it has only one row
# if len(test_gdf) > 1:
#     test_gdf = test_gdf.iloc[[0]].copy()

# # Displaying the resulting test_gdf
# print(test_gdf)

# # Extract multipolygon geometry from the GeoDataFrame
# multipolygon_test = test_gdf.iloc[0]['cycling_5_minutes']

# # Create an empty list to store rows with overlaps
# overlapping_rows = []

# # Check overlaps with each geometry in cbs_geo_df
# for index, row in cbs_geo_df.iterrows():
#     if multipolygon_test.intersects(row['geometry']):
#         print(f"Overlap with index {index}, postcode {row['postcode']}")
#         row_copy = row.copy()
        
#         # Calculate the intersection area
#         intersection_area = multipolygon_test.intersection(row['geometry']).area
        
#         # Calculate percentage overlap relative to postcode geometry
#         row_copy['percentage_overlap'] = (intersection_area / row['geometry'].area) * 100
        
#         overlapping_rows.append(row_copy)

# # Create a GeoDataFrame from the overlapping rows
# overlapping_gdf = gpd.GeoDataFrame(overlapping_rows, 
#                                    crs=cbs_geo_df.crs,
#                                    geometry=[row['geometry'] for row in overlapping_rows])

# # Plot the geometries with overlaps
# fig, ax = plt.subplots(figsize=(10, 10))
# overlapping_gdf.plot(ax=ax, color='navy', edgecolor='white')

# # Annotate with postcodes and percentage overlap
# for idx, row in overlapping_gdf.iterrows():
#     centroid = row['geometry'].centroid
#     text = f"{row['postcode']}\n{row['percentage_overlap']:.2f}%"
#     ax.text(centroid.x, centroid.y, text, fontsize=12, color='white', ha='center', va='center')

# # Plot the original multipolygon_test with thick white outline and transparent white fill
# test_gdf.set_geometry('cycling_5_minutes').plot(ax=ax, facecolor='none', edgecolor='white', alpha=0.5, linewidth=3)

# # Set plot title and show plot
# plt.title('Overlapping Geometries')
# plt.show()



















