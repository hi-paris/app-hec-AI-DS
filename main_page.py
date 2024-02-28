import os
import streamlit as st
import pandas as pd
import numpy as np

from st_pages import Page, show_pages
from PIL import Image
#from utils import authenticate_drive



##################################################################################
#                              PAGE CONFIGURATION                                #
##################################################################################

st.set_page_config(layout="wide")




##################################################################################
#                              GOOGLE DRIVE CONNEXION                            #
##################################################################################

# if ["drive_oauth"] not in st.session_state:
#     st.session_state["drive_oauth"] = authenticate_drive()

# drive_oauth = st.session_state["drive_oauth"]




##################################################################################
#                                   TITLE                                        #
##################################################################################

st.image("images/AI.jpg")
st.title("AI and Data Science Examples")
st.subheader("HEC Paris, 2023-2024")
st.markdown("Course provided by **Shirish C. SRIVASTAVA**")

st.markdown(" ")
st.info("""**About the app**: The AI and Data Science Examples app was created to introduce students to the field of Data Science by showcasing real-life applications of AI.
        It includes use cases using traditional Machine Learning algorithms on structured data, as well as Deep Learning models run on unstructured data (text, images,...).""")

st.divider()


#Hi! PARIS collaboration mention
st.markdown("  ")
image_hiparis = Image.open('images/hi-paris.png')
st.image(image_hiparis, width=150)
url = "https://www.hi-paris.fr/"
st.markdown("**The app was made in collaboration with: [Hi! PARIS Engineering Team](%s)**" % url)




##################################################################################
#                              DASHBOARD/SIDEBAR                                 #
##################################################################################


# AI use case pages
show_pages(
    [
        Page("main_page.py", "Home Page", "🏠"),
        Page("pages/supervised_unsupervised_page.py", "Supervised vs Unsupervised", "🔍"), 
        Page("pages/timeseries_analysis.py", "Time Series Forecasting", "📈"),
        Page("pages/sentiment_analysis.py", "Sentiment Analysis", "👍"),
        #Page("pages/object_detection.py", "Object Detection", "📹"), #need to reduce RAM costs
        Page("pages/recommendation_system.py", "Recommendation system", "🛒")
    ]
)



##################################################################################
#                               PAGE CONTENT                                     #
##################################################################################




