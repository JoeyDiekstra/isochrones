# README

## Overview
This script installs necessary Python packages, reads input data, defines functions to interact with the TravelTime API, and generates isochrones for a set of locations. The results are saved as GeoPackage files.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Functions](#functions)
4. [Output](#output)

## Installation
Ensure you have Python and pip installed. The script will automatically install the required packages if they are not already installed.

## Usage
1. **Change Directory**: Set the working directory where your data and code reside.

2. **Read Input File**: Load the input Excel file.

3. **Environment Variables**: Ensure you have an `.env` file with your `X_APPLICATION_ID` and `X_API_KEY`.

4. **Generate Isochrones**: Use the `get_isochrones` function to generate isochrones.

5. **Save Output**: Save the resulting GeoDataFrames to GeoPackage files in the specified output directory.

## Functions
### `get_recent_monday_noon()`
Returns the most recent Monday at noon in ISO 8601 format.

### `create_payload(lat, lng, travel_time, transport_type)`
Creates a payload for the TravelTime API request.
- **lat** (float): Latitude.
- **lng** (float): Longitude.
- **travel_time** (int): Travel time in seconds.
- **transport_type** (str): Type of transport.

### `get_isochrones(df, travel_times, transport_type, lat_col, lon_col, id_col)`
Generates isochrones for a given set of locations using the TravelTime API.
- **df**: Input data frame.
- **travel_times**: List of travel times in minutes.
- **transport_type**: Type of transport.
- **lat_col**: Column name for latitude.
- **lon_col**: Column name for longitude.
- **id_col**: Unique identifier column.

### `plot_transparent_layers(geo_dfs)`
Plots each GeoDataFrame as transparent layers with different colors for visualization.

## Output
The script saves the generated GeoDataFrames as GeoPackage files in the specified output directory. Each file is named with a timestamp and travel time label.

Example:<your_output_location>/20230913_123456_cycling_5_minutes_output.gpkg
