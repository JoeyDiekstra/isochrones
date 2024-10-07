# -------------------------------------------------------------------------------------------------------------------------------------
# Import packages
# -------------------------------------------------------------------------------------------------------------------------------------

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
from shapely import wkt

# ---------------------------------------------------------------------------
# Introduce page
# ---------------------------------------------------------------------------

# Title and Description
st.title("Isochrone Amenities Analysis")
st.markdown("""
This application calculates the availability and proximity of various amenities (e.g., schools, hospitals, shops) 
within each isochrone generated in the previous step.
""")


## -------------------------------------------------------------------------------------------------------------------------------------
## Read data
## -------------------------------------------------------------------------------------------------------------------------------------

os.chdir(r"C:\Users\Joey.Diekstra\OneDrive - OC&C Strategy Consultants\Personal\python\location_analytics\data\Locatus") # Example

# Read the xlsx file
locatus_data = pd.read_excel("Locatus VKP export.xlsx")

# Display the first few rows of the dataframe to verify
print(locatus_data.head())

# Convert the WKT column to geometries
locatus_data['geometry'] = locatus_data['WKT'].apply(wkt.loads)

# Create latitude and longitude columns by extracting the x and y coordinates from the 'geometry' column
locatus_data['longitude'] = locatus_data['geometry'].apply(lambda geom: geom.x if geom else None)
locatus_data['latitude'] = locatus_data['geometry'].apply(lambda geom: geom.y if geom else None)

locatus_data.rename(columns={'Code': 'uid'}, inplace=True)

# Write the locatus_data DataFrame to an Excel file with the specified name
locatus_data.to_excel("Locatus_data_2023_inc_latlon.xlsx", index=False)

# Read the xlsx file
locatus_data = pd.read_excel("Locatus_data_2023_inc_latlon.xlsx")

locatus_data['Name'].unique()
locatus_data[locatus_data['Name'].isin(['Albert Heijn', 'Jumbo'])]

# Creating a new column 'test_type' and ensuring that only string values are processed
locatus_data['test_type'] = locatus_data['Name'].apply(
    lambda x: 'supermarket' if isinstance(x, str) and ('Albert Heijn' in x or 'Jumbo' in x) else 'other'
)

# Display the first few rows to confirm the new column
print(locatus_data[['Name', 'test_type']].head())


# Write the locatus_data DataFrame to an Excel file with the specified name
locatus_data.to_excel("JD_test_locatus_2023.xlsx", index=False)
