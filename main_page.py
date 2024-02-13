import base64
import os
import altair as alt
import streamlit as st
import pandas as pd

from st_pages import Page, show_pages, add_page_title
from pandas.plotting import register_matplotlib_converters
from pathlib import Path
from PIL import Image
from htbuilder import HtmlElement, div, hr, a, p, img, styles
from htbuilder.units import percent, px

register_matplotlib_converters()


################################################### App configuration ################################################################

st.set_page_config(
    page_title="Introduction to Data Science", layout="wide", page_icon="./images/flask.png"
)

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)

@st.cache_data # compression data
def get_data():
    source = data.stocks()
    source = source[source.date.gt("2004-01-01")]
    return source


@st.cache_data
def get_chart(data):
    hover = alt.selection_single(
        fields=["Date_2"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    lines = (
        alt.Chart(data, title="Evolution of stock prices")
        .mark_line()
        .encode(
            x="Date_2",
            y=kpi,
            #color="symbol",
            # strokeDash="symbol",
        )
    )

    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=65)

    # Draw a rule at the location of the selection
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="yearmonthdate(date)",
            y=kpi,
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("date", title="Date"),
                alt.Tooltip(kpi, title="Price (USD)"),
            ],
        )
        .add_selection(hover)
    )

    return (lines + points + tooltips).interactive()


@st.cache_data
def convert_df(df):
# IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')



##################################################################################
#################################### PASSWORD ####################################
##################################################################################

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if "password" in st.session_state and st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True


# if check_password():
#     # st.write("Here goes your normal Streamlit app...")
#     # st.button("Click me")



# ########### TITLE #############

st.image("images/AI.jpg")
st.title("AI and Data Science Examples")
st.subheader("HEC Paris, 2023-2024")
st.markdown("Course provided by **Shirish C. SRIVASTAVA**")
#st.image("images/hec.png", width=300)
st.markdown("---")



##################################################################################
#                              DASHBOARD/SIDEBAR                                 #
##################################################################################


# AI use case pages
show_pages(
    [
        Page("main_page.py", "Home", "🏠"),
        Page("pages/timeseries_analysis.py", "Time Series Forecasting", "📈"),
        Page("pages/sentiment_analysis.py", "Sentiment Analysis", "🤔"),
        #Page("pages/image_classification.py", "Image classification", ":camera:"),
        Page("pages/object_detection.py", "Object Detection", "📹"), #need to reduce RAM costs
        Page("pages/recommendation_system.py", "Recommendation system", "🛒")
    ]
)


# Hi! PARIS collaboration mention
# st.markdown("  ")
# image_hiparis = Image.open('images/hi-paris.png')
# st.image(image_hiparis, width=150)
# url = "https://www.hi-paris.fr/"
# st.markdown("**The app was made in collaboration with: [Hi! PARIS Engineering Team](%s)**" % url)





##################################################################################
#                               PAGE CONTENT                                     #
##################################################################################

st.markdown("## Introduction to Data Science")


st.divider()

st.markdown("## Supervised vs Unsupervised Learning")
st.markdown("""In the field of Data Science, there are two main approaches/frameworks to build algorithms, **Supervised** and **Unsupervised learning**. 
            Being able to distinguish which type of model fits your application is essential step in building an AI project.
- In **Supervised Learning**, models are trained by learning from "labeled data", which is data that contains the expected value to predict. Labeled data provides to the model the desired output, which it will then use to learn patterns and make predictions.   
- In **Unsupervised Learning**, models learn the data's inherent structure without any specific guidance or instruction. The algorithm will identify any naturally occurring patterns in the dataset using "unlabeled data".
""")

path_ML_data = r"data/ML"


ml_usecase = st.selectbox("Choose an application", 
                          ["Supervised Learning", 
                           "Unsupervised Learning for Customer Segmentation"])

# if ml_usecase == "Unsupervised Learning for Customer Segmentation":
#     customer_data = pd.read_csv(os.path.join(path_ML_data, "customer_segmentation.csv"))
#     customer_data.drop(columns=["ID","Var_1","Segmentation"], inplace=True)
#     st.dataframe(customer_data)