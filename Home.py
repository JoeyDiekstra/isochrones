# # Run Streamlit in powershell
# cd "C:\Users\Joey.Diekstra\OneDrive - OC&C Strategy Consultants\Personal\python\location_analytics\isochrones"
# streamlit run Home.py

import streamlit as st

# Introduction Page
st.title("Welcome to the Isochrones Tool")
st.write("""
    Isochrones are lines on a map that connect points at which something occurs or arrives at the same time. This tool provides two main functionalities:

    1. **Isochrone Generator**: Allows you to generate isochrones for different locations and parameters.
    2. **Isochrone Analysis**: Enables detailed analysis of the generated isochrones.

    Please refer to the specific sections in the documentation for more details on how to use these tools effectively.
""")

# Displaying a moving image (GIF) of cool isochrones
st.image("traveltime_gif.gif", caption="Moving isochrone example")


