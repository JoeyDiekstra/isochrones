## -------------------------------------------------------------------------------------------------------------------------------------
## Import packages
## -------------------------------------------------------------------------------------------------------------------------------------

# # Run Streamlit in powershell
# cd "C:\Users\Joey.Diekstra\OneDrive - OC&C Strategy Consultants\Personal\python\location_analytics\isochrones"
# streamlit run streamlit_isochrone_generator.py

# Now import the required modules
import streamlit as st
import os
import time
import pandas as pd
import geopandas as gpd
import requests
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from dotenv import load_dotenv, find_dotenv
from datetime import datetime, timedelta

## -------------------------------------------------------------------------------------------------------------------------------------
## Read data
## -------------------------------------------------------------------------------------------------------------------------------------

# os.chdir(r"C:\Users\Joey.Diekstra\OneDrive - OC&C Strategy Consultants\Personal\python\location_analytics\data") # Example

# Title and Description
st.title("Isochrone Generator")
st.markdown("""
**Welcome to the Isochrone Generator App!**
This application allows you to generate isochrones, which are areas of equal travel time to a specific point.
You can specify various inputs and see the results in real-time as you interact with the app.
""")

# Description of required columns
st.markdown("""
Please upload an Excel file with the following columns:
- **1. Unique Identifier**: A column with any data type.
- **2. Latitude**: A column with numerical data type.
- **3. Longitude**: A column with numerical data type.
""")

# File uploader prompt
uploaded_file = st.file_uploader(
    "Upload the Excel file here",
    type=["xlsx"]
)

if uploaded_file is not None:
    try:
        # Read the Excel file
        input_df = pd.read_excel(uploaded_file)

        input_df = input_df[:1]  # delete in final code

        # Display the DataFrame (optional)
        st.write(input_df.head())

        # Conditionally write success message
        st.success('File successfully imported')

    except Exception as e:
        # Handle other potential exceptions
        st.error(f'An unexpected error occurred: {e}')


# ## -------------------------------------------------------------------------------------------------------------------------------------
# ## Define functions
# ## -------------------------------------------------------------------------------------------------------------------------------------

def get_recent_monday_noon():
    """
    Get the most recent Monday at noon in UTC time.

    Returns:
        str: The most recent Monday at noon in ISO 8601 format.
    """
    now = datetime.utcnow()
    # Calculate the number of days since last Monday
    days_since_monday = (now.weekday() - 0) % 7
    # Find the most recent Monday
    recent_monday = now - timedelta(days=days_since_monday)
    # Set the time to noon
    recent_monday_noon = recent_monday.replace(hour=12, minute=0, second=0, microsecond=0)
    return recent_monday_noon.isoformat() + "Z"

def create_payload(lat, lng, travel_time, transport_type):
    """
    Create a payload for the TravelTime API request.

    Args:
        lat (float): The latitude of the location.
        lng (float): The longitude of the location.
        travel_time (int): The travel time in seconds.
        transport_type (str): Type of transport (e.g., 'cycling', 'driving', 'public_transport', 'walking', 'coach', 'bus', 'train').

    Returns:
        dict: A dictionary representing the payload for the API request.
    
    Raises:
        ValueError: If any input parameters are of invalid type.
    """
    # Validate input types
    if not isinstance(lat, (int, float)):
        raise ValueError("Latitude must be a number.")
    if not isinstance(lng, (int, float)):
        raise ValueError("Longitude must be a number.")
    if not isinstance(travel_time, int):
        raise ValueError("Travel time must be an integer.")
    if not isinstance(transport_type, str):
        raise ValueError("Transport type must be a string.")

    arrival_time = get_recent_monday_noon()

    return {
        "arrival_searches": [
            {
                "id": f"{transport_type}_{travel_time}_seconds",
                "coords": {
                    "lat": lat,
                    "lng": lng
                },
                "arrival_time": arrival_time,
                "travel_time": travel_time,
                "transportation": {
                    "type": transport_type
                },
                "level_of_detail": {        
                    "scale_type": "simple",        
                    "level": "medium"      
                },
                "range": {
                    "enabled": True,
                    "width": 3600
                }
            }
        ]
    }

def get_isochrones(df, travel_times, transport_type, lat_col, lon_col, id_col):
    """
    Generate isochrones for a given set of locations using the TravelTime API.

    Args:
        df (DataFrame): Input data frame containing locations.
        travel_times (list of int): List of travel times in minutes.
        transport_type (str): Type of transport (e.g., 'cycling', 'driving', 'public_transport', 'walking', 'coach', 'bus', 'train').
        lat_col (str): Name of the column for latitude.
        lon_col (str): Name of the column for longitude.
        id_col (str): Name of the unique identifier column.

    Returns:
        dict of GeoDataFrames: A dictionary where keys are travel time labels and values 
                               are GeoDataFrames containing polygons for each travel time.
    """

    travel_times = [tt * 60 for tt in travel_times]
    geo_dfs = {f'{transport_type}_{tt//60}_minutes': [] for tt in travel_times}
    total_iterations = len(df) * len(travel_times)
    start_time = time.time()
    
    # Initialize progress bar
    progress_bar = st.progress(0)

    current_iteration = 0
    call_count = 0
    window_start_time = time.time()

    for index, row in df.iterrows():
        for tt in travel_times:
            payload = create_payload(row[lat_col], row[lon_col], tt, transport_type)
            response = requests.post('https://api.traveltimeapp.com/v4/time-map', headers=headers, json=payload)
            
            if response.status_code == 200:
                response_json = response.json()
                multipolygon = []
                
                for result in response_json['results']:
                    for shape in result['shapes']:
                        shell_points = [(point['lng'], point['lat']) for point in shape['shell']]
                        holes = [[(hole_point['lng'], hole_point['lat']) for hole_point in hole] for hole in shape.get('holes', [])]

                        polygon = Polygon(shell=shell_points, holes=holes)
                        multipolygon.append(polygon)
                
                multi_polygon_obj = MultiPolygon(multipolygon)
                col_name = f'{transport_type}_{tt//60}_minutes'
                geo_dfs[col_name].append({
                    id_col: row[id_col],
                    lat_col: row[lat_col],
                    lon_col: row[lon_col],
                    'geometry': multi_polygon_obj
                })
                
            else:
                st.error(f"Request failed with status code: {response.status_code}")
                st.error(response.text)
            
            # Update the progress bar
            current_iteration += 1
            progress_bar.progress(current_iteration / total_iterations)
            
            # Increment the API call count
            call_count += 1
            
            # If we've made 5 calls, ensure we wait before making more
            if call_count == 5:
                elapsed = time.time() - window_start_time
                if elapsed < 60:
                    time.sleep(60 - elapsed)
                call_count = 0
                window_start_time = time.time()

    end_time = time.time()
    elapsed_time = end_time - start_time
    st.write(f"Total process took {elapsed_time:.2f} seconds.")
    
    for key in geo_dfs.keys():
        geo_dfs[key] = gpd.GeoDataFrame(geo_dfs[key], crs="EPSG:4326")
    
    return geo_dfs

def plot_transparent_layers(geo_dfs):
    """
    Plots each GeoDataFrame in `geo_dfs` as transparent layers with beautiful colors.

    Parameters:
        geo_dfs (dict): A dictionary where keys are labels and values are GeoDataFrames.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Defining color palette and labels
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    labels = list(geo_dfs.keys())

    for i, (key, gdf) in enumerate(geo_dfs.items()):
        gdf.plot(ax=ax, color=colors[i], alpha=0.5, edgecolor='k', label=f'{labels[i]}')

    # Adding a legend
    plt.legend(title="Travel Time", loc='upper left')

    # Setting title and axis labels
    plt.title("Isochrones for Travel Times", fontsize=15)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Adding grid for better readability
    plt.grid(True)

    # Displaying the plot in Streamlit
    st.pyplot(fig)

# ## -------------------------------------------------------------------------------------------------------------------------------------
# ## Generate data
# ## -------------------------------------------------------------------------------------------------------------------------------------

# Allow the user to upload an .env file
uploaded_env_file = st.file_uploader("Upload a .env file that contains the TravelTime API credentials", type=["env"])

if uploaded_env_file is not None:
    try:
        # Save the uploaded .env file to the current directory
        with open(".env", "wb") as f:
            f.write(uploaded_env_file.getbuffer())

        # Load the .env file
        if load_dotenv(find_dotenv()):
            # Write TRUE if .env file is successfully loaded
            st.write("Environment variables have been loaded successfully")
        else:
            st.error("Failed to load .env file. Please check the file.")

    except Exception as e:
        # Handle any exceptions that occur
        st.error(f'An error occurred: {e}')


# Retrieve the variables
application_id = os.getenv('X_APPLICATION_ID')
api_key = os.getenv('X_API_KEY')

# Define headers for the API request (replace with your actual headers)
headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'X-Application-Id': application_id,
    'X-Api-Key': api_key
}


# Initialize session_state variables if they don't exist
if 'geo_dfs' not in st.session_state:
    st.session_state.geo_dfs = None

if 'output_dir_set' not in st.session_state:
    st.session_state.output_dir_set = False

if 'process_started' not in st.session_state:
    st.session_state.process_started = False

# Step 1: Collect user inputs for isochrone generation
user_travel_times = st.text_input("Enter travel times in minutes (e.g., 10, 20, 30)").split(',')
travel_times = [int(x.strip()) for x in user_travel_times if x.strip().isdigit() and 1 <= int(x.strip()) <= 240]

transport_options = ["cycling", "driving", "public_transport", "walking"]
transport_type = st.selectbox("Select transport type", transport_options)

latitude_column = st.text_input("Enter the latitude column name")

longitude_column = st.text_input("Enter the longitude column name")

unique_identifier = st.text_input("Enter the unique identifier column name (e.g., uid)")

if st.button('Generate Isochrones') and not st.session_state.process_started:
    st.session_state.process_started = True
    
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    for i in range(101):
        progress_bar.progress(i)
        progress_text.text(f"Progress: {i}%")
    
    geo_dfs = get_isochrones(
        df=input_df,
        travel_times=travel_times,
        transport_type=transport_type,
        lat_col=latitude_column.strip(),
        lon_col=longitude_column.strip(),
        id_col=unique_identifier.strip()
    )

    plot_transparent_layers(geo_dfs)
    
    st.session_state.geo_dfs = geo_dfs
    st.success("Isochrones generated successfully.")

# Step 2: Specify location to save generated output
if st.session_state.geo_dfs and not st.session_state.output_dir_set:
    output_dir = st.text_input("Specify the output directory to save the GeoPackage files")
    
    if output_dir and st.button('Save Output'):
        try:
            current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if not os.path.exists(rf"{output_dir}"):
                os.makedirs(rf"{output_dir}")
                
            for key, gdf in st.session_state.geo_dfs.items():
                gpkg_path = os.path.join(rf"{output_dir}", f"{current_datetime}_{key}_output.gpkg")
                gdf.to_file(gpkg_path, layer=key, driver='GPKG')
                st.write(f"GeoPackage for {key} has been created at: {gpkg_path}")
            
            st.success("All GeoPackages have been created successfully.")
            st.session_state.output_dir_set = True
        
        except Exception as e:
            st.error(f"An error occurred while saving the files: {e}")


# Initialize session state variables if they don't exist
if 'output_dir_set' not in st.session_state:
    st.session_state.output_dir_set = False

# Define a function to clear all session state variables
def clear_session_state():
    for key in st.session_state.keys():
        st.session_state[key] = None

# Step 3: Provide option to start new data generation if output directory is set
if st.session_state.output_dir_set:
    if st.button("Generate New Data"):
        # Clear session state variables
        clear_session_state()
        # Optionally, reset other input fields if needed
        st.rerun()

