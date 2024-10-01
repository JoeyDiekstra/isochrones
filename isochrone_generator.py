## -------------------------------------------------------------------------------------------------------------------------------------
## Import packages
## -------------------------------------------------------------------------------------------------------------------------------------

import subprocess
import sys

def install(package):
    """Installs the package using conda."""
    try:
        subprocess.check_call(["conda", "install", "--yes", package])
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}. Trying to install it via pip.")
        subprocess.check_call([sys.executable, "-m", "conda", "install", package])

# List of packages to ensure they are installed
required_packages = [
    "pandas", "geopandas", "requests", "matplotlib",
    "shapely", "tqdm", "python-dotenv", "streamlit"
]

# Ensure all required packages are installed
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install(package)

# Now import the required modules
import streamlit as st
import os
import time
import pandas as pd
import geopandas as gpd
import requests
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm  # Import tqdm for progress bar
from shapely import wkt
from dotenv import load_dotenv
from datetime import datetime, timedelta

## -------------------------------------------------------------------------------------------------------------------------------------
## Read data
## -------------------------------------------------------------------------------------------------------------------------------------

# Change the current working directory to the specified path where the data and code reside.
os.chdir("<YOUR_FILE_LOCATION>")
os.chdir(r"C:\Users\Joey.Diekstra\OneDrive - OC&C Strategy Consultants\Personal\python\location_analytics\data") # Example

# List all files in the current directory
files = [f for f in os.listdir() if os.path.isfile(f)]

# Print each file
for file in files:
    print(file)

# Read input file
input_df = pd.read_excel("20240930 example addresses.xlsx")

input_df = input_df[:1]

# Display the DataFrame
print(input_df)

## -------------------------------------------------------------------------------------------------------------------------------------
## Define functions
## -------------------------------------------------------------------------------------------------------------------------------------

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
    # Convert travel times from minutes to seconds
    travel_times = [tt * 60 for tt in travel_times]
    # Initialize lists for each travel time
    geo_dfs = {f'{transport_type}_{tt//60}_minutes': [] for tt in travel_times}
    
    # Total number of iterations = number of rows * number of travel times
    total_iterations = len(df) * len(travel_times)
    # Create a tqdm progress bar
    with tqdm(total=total_iterations, desc="Processing Isochrones") as pbar:
        # Record the start time
        start_time = time.time()
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
                    print(f"Request failed with status code: {response.status_code}")
                    print(response.text)
                
                # Sleep to ensure we make a maximum of 5 requests per minute
                time.sleep(12)

                # Update progress bar
                pbar.update(1)

        
        # Record the end time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total process took {elapsed_time:.2f} seconds.")
    
    # Convert lists to GeoDataFrames
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

    # Displaying the plot
    plt.show()

## -------------------------------------------------------------------------------------------------------------------------------------
## Generate data
## -------------------------------------------------------------------------------------------------------------------------------------

# Load the .env file
load_dotenv(r"C:\Users\Joey.Diekstra\OneDrive - OC&C Strategy Consultants\Personal\python\location_analytics\isochrones\.env")


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

# API POST request
geo_dfs = get_isochrones(
    df=input_df, 
    travel_times=[5, 10], 
    transport_type='cycling', 
    lat_col='latitude', 
    lon_col='longitude', 
    id_col='uid'
)

# Print the resulting GeoDataFrames
for key, gdf in geo_dfs.items():
    print(f"\nGeoDataFrame: {key}")
    print(gdf.shape)
    print(gdf.head())


# # Plot geo dataframes (no requirement; just for visual purposes)
# plot_transparent_layers(geo_dfs)


## -------------------------------------------------------------------------------------------------------------------------------------
## Write data
## -------------------------------------------------------------------------------------------------------------------------------------

# Define datetime variable
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define output directory
output_dir = r"<YOUR\OUTPUT\LOCATION>"
# output_dir = r"C:\Users\Joey.Diekstra\OneDrive - OC&C Strategy Consultants\Personal\python\location_analytics\output" # Example

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# # Write each GeoDataFrame to a Shapefile
# for key, gdf in geo_dfs.items():
#     file_path = f"{output_dir}\{current_datetime}_{key}.shp"
#     gdf.to_file(file_path, driver='ESRI Shapefile')

# Write each GeoDataFrame to a separate GeoPackage file
for key, gdf in geo_dfs.items():
    gpkg_path = os.path.join(output_dir, f"{current_datetime}_{key}_output.gpkg")
    gdf.to_file(gpkg_path, layer=key, driver='GPKG')
    print(f"GeoPackage for {key} has been created at: {gpkg_path}")

print("All GeoPackages have been created")
