# # Run Streamlit in powershell
# cd "C:\Users\Joey.Diekstra\OneDrive - OC&C Strategy Consultants\Personal\python\location_analytics\isochrones"
# streamlit run Home.py

import streamlit as st

# Introduction Page
st.title("Catchment Analysis Tool")
st.write("""
    The Catchment Analysis Tool is based on the usage of isochrones. Isochrones are lines on a map that connect points at which something occurs or arrives at the same time. This tool provides two main functionalities:

    1. **Isochrone Generator**: Allows you to generate isochrones for different locations and parameters.
    2. **Isochrone Population Analysis**: Allows you to calculate the population, segmented by age group, along with additional demographic data for each of the generated isochrones.
    3. **Isochrone Amenities Analysis**: Enables you to calculate the number of ammenities (e.g., schools, supermarkets, shops) for each of the generated isochrones.

    Please refer to the specific sections in the documentation for more details on how to use these tools effectively.
""")

# Displaying a moving image (GIF) of cool isochrones
st.image("traveltime_gif.gif", caption="Example of isochrones")


