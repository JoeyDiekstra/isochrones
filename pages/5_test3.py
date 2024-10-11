# ---------------------------------------------------------------------------
# Import packages
# ---------------------------------------------------------------------------

import os
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import time
import streamlit as st
import re

# Mapping
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static


os.chdir(r"C:\Users\Joey.Diekstra\OneDrive - OC&C Strategy Consultants\Personal\python\location_analytics\output")

df = pd.read_excel("20241011_121427_cycling_12_minutes_output.xlsx")

# ---------------------------------------------------------------------------
# Introduce page
# ---------------------------------------------------------------------------

# Introduce page
st.title("Isochrone Amenities Analysis")
st.markdown("""
This tool calculates the availability of various amenities (e.g., schools, supermarkets, shops) within each isochrone generated in step 1. 
For each isochrone, it provides a count of the different types of amenities.
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
    "Upload the CSV file(s) generated in step 1 here:",
    type=["csv"],
    accept_multiple_files=True
)

st.markdown("""
Please upload the Excel file containing the following columns:
- **1. Latitude**: A column with numerical data type.
- **2. Longitude**: A column with numerical data type.
- **3. Amenity Type**: A column containing the type of amenity, allowing for categorisation and counting by type.
""")

uploaded_excel_file = st.file_uploader(
    "Upload the Excel file here",
    type=["xlsx"]
)

latitude_column = st.text_input("Enter the latitude column name")
longitude_column = st.text_input("Enter the longitude column name")
amenity_type = st.text_input("Enter the amenity type column name (e.g., type)")

# Button to read data
if st.button('Read Data') and uploaded_files and uploaded_excel_file and all([latitude_column, longitude_column, amenity_type]):
    try:
        # Show reading message
        msg_placeholder = st.empty()
        msg_placeholder.info("Reading data...")

        # Read and validate GeoDataFrames
        geo_dfs = {}
        gdf_names_dict = {}

        import geopandas as gpd
        import pandas as pd
        from shapely import wkt
        from shapely.geometry import MultiPolygon
        from shapely import wkt
        import logging

        def safe_wkt_loads(geom):
            try:
                # Attempt to load WKT
                return wkt.loads(geom)
            except Exception as e:
                # Log detailed information for debugging
                logging.error(f"Failed to parse WKT: {geom[:100]}... Error: {e}")
                # You can choose to return None or handle it differently
                return None

        
        # Assuming uploaded_files is a list of uploaded XLSX files.
        
        for uploaded_file in uploaded_files:
            new_file_name = generate_and_validate_filename(uploaded_file.name)
            
            # Read the Excel file into a Pandas DataFrame
            df = pd.read_csv(uploaded_file)
        
            if 'geometry' in df.columns:
                # Convert the 'geometry' column from WKT strings to actual geometries
                df['geometry'] = df['geometry'].apply(safe_wkt_loads)
                
                # Convert the DataFrame to a GeoDataFrame
                gdf = gpd.GeoDataFrame(df, geometry='geometry')
                gdf = gdf.rename(columns={'geometry': new_file_name})
                
                # Correctly assign the geometry column after renaming
                gdf.set_geometry(new_file_name, inplace=True)

                # Set the Coordinate Reference System (CRS)
                gdf.set_crs('EPSG:4326', inplace=True)
                
                # Validate and potentially rename the geometry column
                if new_file_name not in gdf.columns or not gdf.is_valid.all():
                    raise ValueError(f"The geometry column in the file '{uploaded_file.name}' is not valid.")
                
                # Set the Coordinate Reference System (CRS)
                # gdf.set_crs('EPSG:4326', inplace=True)
                
                # Store the GeoDataFrame in a dictionary with its name
                geo_dfs[new_file_name] = gdf
                gdf_names_dict[new_file_name] = uploaded_file.name
            else:
                raise ValueError(f"No 'geometry' column found in the file '{uploaded_file.name}'.")
        
        # Read input Excel file
        input_df = pd.read_excel(uploaded_excel_file)

        # Create a new column 'geometry' in input_df by applying a function to each row.
        # The function takes two arguments, longitude and latitude, and uses them to create a Point object.
        input_df['geometry'] = input_df.apply(lambda row: Point(row[longitude_column], row[latitude_column]), axis=1)

        # Convert the DataFrame into a GeoDataFrame
        # This assigns the new 'geometry' column as the geometry for the GeoDataFrame
        # Set the coordinate reference system (CRS) to "EPSG:4326", which corresponds to WGS84 (latitude/longitude)
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
        last_isochrone_geometry = None  # Variable to store the last geometry

        for i, (key, gdf) in enumerate(st.session_state.geo_dfs.items()):
            gdf['transport_method_time'] = key
            temp_input_gdf = st.session_state.input_gdf.copy()
            temp_input_gdf['overlap'] = False

            for test_type in unique_test_types:
                gdf[test_type] = 0

            for row_index, row in gdf.iterrows():
                row_geometry = row[key]
                temp_input_gdf['overlap'] = temp_input_gdf['geometry'].apply(lambda x: row_geometry.intersects(x))

                for test_type in unique_test_types:
                    true_count = temp_input_gdf[(temp_input_gdf[amenity_type] == test_type) & (temp_input_gdf['overlap'])].shape[0]
                    gdf.at[row_index, test_type] = true_count

            df_list.append(gdf.drop(columns=[key, latitude_column, longitude_column]))

            # Update last isochrone geometry to the last row's geometry in the current gdf
            last_isochrone_geometry = gdf.iloc[-1][key]

            progress_bar.progress((i + 1) / total_steps)  # Update progress

        final_df = pd.concat(df_list, ignore_index=True)
        
        # Store the final DataFrame in session state
        st.session_state.final_df = final_df
        
        calculation_time = time.time() - start_time

        # Calculation result message
        calc_msg_placeholder.success(f"Amenities calculated successfully in {calculation_time:.2f} seconds.")

        # Set calculate_amenities to True
        st.session_state.calculate_amenities = True
        
        # Plot the last isochrone with overlapping points
        if last_isochrone_geometry is not None:
            # Filter out the overlapping points
            overlapping_points = st.session_state.input_gdf[
                st.session_state.input_gdf['geometry'].apply(lambda x: last_isochrone_geometry.intersects(x))
            ]

            # Generate and display the map
            center = (last_isochrone_geometry.centroid.y, last_isochrone_geometry.centroid.x)
            m = plot_isochrone_with_amenities(
                gpd.GeoDataFrame({'geometry': [last_isochrone_geometry]}),
                overlapping_points,
                center
            )

            folium_static(m)  # Use folium_static to display in Streamlit

            # Show caption
            st.caption("An example of an isochrone analysed along with its corresponding amenities")


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
