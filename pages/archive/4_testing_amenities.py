## -------------------------------------------------------------------------------------------------------------------------------------
## Import packages
## -------------------------------------------------------------------------------------------------------------------------------------

# import os
# import pandas as pd
# from shapely.geometry import Point
# import geopandas as gpd
# import matplotlib.pyplot as plt
# import time
# from tqdm import tqdm
# from datetime import datetime
# import re
# import streamlit as st
# from shapely import wkt

# # ---------------------------------------------------------------------------
# # Introduce page
# # ---------------------------------------------------------------------------

# # Title and Description
# st.title("Isochrone Amenities Analysis")
# st.markdown("""
# This application calculates the availability and proximity of various amenities (e.g., schools, hospitals, shops) 
# within each isochrone generated in the previous step.
# """)


# ## -------------------------------------------------------------------------------------------------------------------------------------
# ## Read data
# ## -------------------------------------------------------------------------------------------------------------------------------------

# # os.chdir(r"C:\Users\Joey.Diekstra\OneDrive - OC&C Strategy Consultants\Personal\python\location_analytics\data\Locatus") # Example

# # # Read the xlsx file
# # locatus_data = pd.read_excel("Locatus VKP export.xlsx")

# # # Display the first few rows of the dataframe to verify
# # print(locatus_data.head())

# # # Convert the WKT column to geometries
# # locatus_data['geometry'] = locatus_data['WKT'].apply(wkt.loads)

# # # Create latitude and longitude columns by extracting the x and y coordinates from the 'geometry' column
# # locatus_data['longitude'] = locatus_data['geometry'].apply(lambda geom: geom.x if geom else None)
# # locatus_data['latitude'] = locatus_data['geometry'].apply(lambda geom: geom.y if geom else None)

# # locatus_data.rename(columns={'Code': 'uid'}, inplace=True)

# # # Write the locatus_data DataFrame to an Excel file with the specified name
# # locatus_data.to_excel("Locatus_data_2023_inc_latlon.xlsx", index=False)

# # # Read the xlsx file
# # locatus_data = pd.read_excel("Locatus_data_2023_inc_latlon.xlsx")

# # locatus_data['Name'].unique()
# # locatus_data[locatus_data['Name'].isin(['Albert Heijn', 'Jumbo'])]

# # # Creating a new column 'test_type' and ensuring that only string values are processed
# # locatus_data['test_type'] = locatus_data['Name'].apply(
# #     lambda x: 'supermarket' if isinstance(x, str) and ('Albert Heijn' in x or 'Jumbo' in x) else 'other'
# # )

# # # Display the first few rows to confirm the new column
# # print(locatus_data[['Name', 'test_type']].head())


# # # Write the locatus_data DataFrame to an Excel file with the specified name
# # locatus_data.to_excel("JD_test_locatus_2023.xlsx", index=False)



# ---------------------------------------------------------------------------
# Import packages
# ---------------------------------------------------------------------------

import os
import pandas as pd
from shapely.geometry import Point
from shapely.strtree import STRtree
import geopandas as gpd
import time
import streamlit as st
import re

# Mapping
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static

# ---------------------------------------------------------------------------
# Introduce page
# ---------------------------------------------------------------------------

# Introduce page
st.title("Isochrone Amenities Analysis")
st.markdown("""
This application calculates the availability and proximity of various amenities (e.g., schools, hospitals, shops) 
within each isochrone generated in the previous step.
""")

# ---------------------------------------------------------------------------
# Define functions
# ---------------------------------------------------------------------------

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


def plot_isochrone_with_amenities(isochrone_gdf, amenities_gdf, start_coords):
    """
    Plots isochrones and amenities on a Folium map.

    Parameters:
    - isochrone_gdf: GeoDataFrame containing isochrone geometries.
    - amenities_gdf: GeoDataFrame with amenities location points.
    - start_coords: Tuple of (latitude, longitude) for initial map centering.
    """

    # Initialize a Folium map centered at the starting coordinates
    m = folium.Map(location=start_coords, tiles='CartoDB positron', zoom_start=13)

    # Add isochrone polygons to the map
    for _, row in isochrone_gdf.iterrows():
        # Drawing each polygon based on the adjusted geometry column name
        geometry_column = isochrone_gdf.geometry.name
        folium.GeoJson(
            data=row[geometry_column],
            style_function=lambda x: {
                'fillColor': 'lightblue',
                'color': 'blue',
                'weight': 2,
                'fillOpacity': 0.5
            }
        ).add_to(m)

    # Add amenities as clustered markers to the map
    marker_cluster = MarkerCluster().add_to(m)
    for _, row in amenities_gdf.iterrows():
        # Use geometry attributes to extract x and y coordinates
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=row.get('name', 'Amenity'),
            icon=folium.Icon(color='green', icon='info-sign')
        ).add_to(marker_cluster)

    return m


def reset_session():
    # Set the reset flag and reload the page
    st.session_state.reset = True

# ---------------------------------------------------------------------------
# Read Data
# ---------------------------------------------------------------------------

# Initialize session state variables
if 'read_data' not in st.session_state:
    st.session_state.read_data = False
if 'calculate_amenities' not in st.session_state:
    st.session_state.calculate_amenities = False

# Step 1: Upload Data Files and Enter Input Fields
uploaded_files = st.file_uploader(
    "Upload the GPKG file(s) here",
    type=["gpkg"],
    accept_multiple_files=True
)

uploaded_excel_file = st.file_uploader(
    "Upload the Excel file here",
    type=["xlsx"]
)

latitude_column = st.text_input("Enter the latitude column name")
longitude_column = st.text_input("Enter the longitude column name")
unique_identifier = st.text_input("Enter the unique identifier column name (e.g., uid)")
amenity_type = st.text_input("Enter the amenity type column name (e.g., type)")

# Button to read data
if st.button('Read Data') and uploaded_files and uploaded_excel_file and all([latitude_column, longitude_column, unique_identifier, amenity_type]):
    try:
        # Show reading message
        msg_placeholder = st.empty()
        msg_placeholder.info("Reading data...")

        # Read and validate GeoDataFrames
        geo_dfs = {}
        gdf_names_dict = {}

        for uploaded_file in uploaded_files:
            new_file_name = generate_and_validate_filename(uploaded_file.name)
            gdf = gpd.read_file(uploaded_file)
            if 'geometry' in gdf.columns:
                gdf = gdf.rename(columns={'geometry': new_file_name})
                gdf = gdf.set_geometry(new_file_name)
            if new_file_name not in gdf.columns or not gpd.GeoSeries(gdf[new_file_name]).is_valid.all():
                raise ValueError(f"The geometry column '{new_file_name}' is not valid in the file '{uploaded_file.name}'.")
            gdf.set_crs('EPSG:4326', inplace=True)
            geo_dfs[new_file_name] = gdf
            gdf_names_dict[new_file_name] = uploaded_file.name

        # Read input Excel file
        input_df = pd.read_excel(uploaded_excel_file)
        input_df['geometry'] = input_df.apply(lambda row: Point(row[longitude_column], row[latitude_column]), axis=1)
        input_gdf = gpd.GeoDataFrame(input_df, geometry='geometry', crs="EPSG:4326")

        # Update session state
        st.session_state.geo_dfs = geo_dfs
        st.session_state.input_gdf = input_gdf
        st.session_state.gdf_names_dict = gdf_names_dict

        # Successful read message
        msg_placeholder.success("Data read successfully")
        st.session_state.read_data = True

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# ---------------------------------------------------------------------------
# Perform calculations
# ---------------------------------------------------------------------------

# Step 2: Calculate Amenities per Isochrone
if st.session_state.read_data:
    if st.button("Calculate Amenities per Isochrone"):
        calc_msg_placeholder = st.empty()
        calc_msg_placeholder.info("Calculating...")

        start_time = time.time()
        
        # Create progress bar
        progress_bar = st.progress(0)
        total_steps = len(st.session_state.geo_dfs)

        df_list = []
        unique_test_types = st.session_state.input_gdf[amenity_type].unique()
        last_isochrone_geometry = None

        # Prepare spatial index for the input geometries
        input_geometries = list(st.session_state.input_gdf['geometry'])
        spatial_index = STRtree(input_geometries)

        for i, (key, gdf) in enumerate(st.session_state.geo_dfs.items()):
            gdf['transport_method_time'] = key
            temp_input_gdf = st.session_state.input_gdf.copy(deep=False)
            temp_input_gdf['overlap'] = False

            for test_type in unique_test_types:
                gdf[test_type] = 0

            for row_index, row in gdf.iterrows():
                row_geometry = row[key]
                
                # Use spatial index to find potentially overlapping geometries
                possible_matches_index = spatial_index.query(row_geometry)
                possible_matches = temp_input_gdf.iloc[possible_matches_index]

                for test_type in unique_test_types:
                    true_count = possible_matches[
                        (possible_matches[amenity_type] == test_type) & 
                        (possible_matches['geometry'].apply(row_geometry.intersects))
                    ].shape[0]
                    gdf.loc[row_index, test_type] = true_count

            gdf[unique_identifier] = gdf.index
            df_list.append(gdf.drop(columns=[key, latitude_column, longitude_column]))

            # Update last isochrone geometry to the last row's geometry in the current gdf
            last_isochrone_geometry = gdf.iloc[-1][key]

            if i % max(1, total_steps // 100) == 0:  # Update less frequently
                progress_bar.progress((i + 1) / total_steps)

        final_df = pd.concat(df_list, ignore_index=True)
        
        # Store the final DataFrame in session state
        st.session_state.final_df = final_df
        
        calculation_time = time.time() - start_time

        # Calculation result message
        calc_msg_placeholder.success(f"Amenities calculated successfully in {calculation_time:.2f} seconds.")

        st.session_state.calculate_amenities = True
        
        if last_isochrone_geometry is not None:
            overlapping_points = st.session_state.input_gdf[
                st.session_state.input_gdf['geometry'].apply(last_isochrone_geometry.intersects)
            ]

            center = (last_isochrone_geometry.centroid.y, last_isochrone_geometry.centroid.x)
            m = plot_isochrone_with_amenities(
                gpd.GeoDataFrame({'geometry': [last_isochrone_geometry]}),
                overlapping_points,
                center
            )
            folium_static(m)


# Reset session state flag
if 'reset' not in st.session_state:
    st.session_state.reset = False

# Step 3: Download Results
if st.session_state.calculate_amenities:
    st.write("Final Concatenated DataFrame:")
    st.write(st.session_state.final_df)

    csv = st.session_state.final_df.to_csv(index=False)
    
    # Use on_click with reset logic
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name='overlap_output.csv',
        mime='text/csv',
        on_click=reset_session  # Set reset flag
    )

# Ensure the "Generate New Data" button also sets the reset flag
if st.session_state.calculate_amenities:
    if st.button("Generate New Data"):
        reset_session()

# Check if reset is needed
if st.session_state.get('reset', False):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    # Navigate home to perform a reset (refresh)
    st.rerun()
