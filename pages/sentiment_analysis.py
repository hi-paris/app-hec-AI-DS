import os
import re
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px

from pysentimiento import create_analyzer

st.set_page_config(layout="wide")

def clean_text(text):
    pattern_punct = r"[^\w\s.',:/]"
    pattern_date = r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'

    text = text.lower()
    text = re.sub(pattern_date, '', text)
    text = re.sub(pattern_punct, '', text)
    text = text.replace("ggg","g")
    text = text.replace("  "," ")

    return text


st.markdown("# Sentiment Analysis")

st.markdown("### What is Sentiment Analysis ?")

st.info("""
    Sentiment analysis is a **Natural Language Processing** (NLP) task that involves determining the sentiment or emotion expressed in a piece of text. 
    It has a wide range of use cases across various industries, as it helps organizations gain insights into the opinions, emotions, and attitudes expressed in text data.""")

st.markdown("Here is an example of Sentiment analysis used to analyze **Customer Satisfaction** for perfums.")
            
st.image("images/sentiment_analysis.png", width=800)

st.markdown(" ")

st.markdown("""
Common applications of Natural Language Processing include:
- **Customer Feedback and Reviews** 💯: Assessing reviews on products or services to understand customer satisfaction and identify areas for improvement.
- **Market Research** 🔍: Analyzing survey responses or online forums to gauge public opinion on products, services, or emerging trends.
- **Financial Market Analysis** 📉: Monitoring financial news, reports, and social media to gauge investor sentiment and predict market trends.
- **Government and Public Policy** 📣: Analyzing public opinion on government policies, initiatives, and political decisions to gauge public sentiment and inform decision-making.        
""")

st.divider()

#sa_pages = ["Starbucks Customer Reviews (Text)", "Tiktok's US Congressional Hearing (Audio)"]
#st.markdown("### Select a use case ")
#use_case = st.selectbox("", sa_pages, label_visibility="collapsed")


st.markdown("### Starbucks Customer Reviews ☕")
st.warning("""In this use case, we are going to analyze the polarity (negative, neutral, positive) of customer reviews by using Sentiment Analysis. 
           You can try the application by using the provided starbucks customer reviews, or by writing your own.""") 
st.image("images/header_starbucks.webp")



tab1_, tab2_ = st.tabs(["Starbucks reviews", "Write a review"])

#st.markdown("The dataset contains the location (state), date, rating, text and images (if provided) for each review.")

with tab1_:
    # LOAD THE DATA
    reviews_df = pd.read_csv("data/sa_data/reviews_data_clean.csv")
    reviews_df["Year"] = reviews_df["Date"].apply(lambda x: x[:4])
    reviews_df.insert(0, "ID", [f"{i}" for i in np.arange(1, len(reviews_df)+1)])

    # FILTER DATA
    st.markdown(" ")

    col1, col2 = st.columns([0.2, 0.8], gap="medium")

    with col1:
        st.markdown("""<b>Filter reviews: </b> <br>
                    You can filter the dataset by Date, State or Rating""", unsafe_allow_html=True)
        
        select_image_box = st.radio("",
        ["Filter by Date (Year)", "Filter by State", "Filter by Rating", "Remove filters"],
        index=None, label_visibility="collapsed")

        if select_image_box == "Filter by Date (Year)":
            selected_date = st.multiselect("Date (Year)", reviews_df["Year"].unique())
            reviews_df = reviews_df.loc[reviews_df["Year"].isin(selected_date)]

        if select_image_box == "Filter by State":
            selected_state = st.multiselect("State", reviews_df["State"].unique())
            reviews_df = reviews_df.loc[reviews_df["State"].isin(selected_state)]

        if select_image_box == "Filter by Rating":
            selected_rating = st.multiselect("Rating", sorted(list(reviews_df["Rating"].dropna().unique())))
            reviews_df = reviews_df.loc[reviews_df["Rating"].isin(selected_rating)]

        if select_image_box == "Remove filters":
            pass

        #st.slider()

        run_model1 = st.button("**Run the model**", type="primary", key="tab1")

    with col2:
    # VIEW DATA
        st.markdown("""<b>View the reviews:</b> <br>
                    The dataset contains the location (State), date, rating, text and images (if provided) for each review.""", 
                    unsafe_allow_html=True)
        
        st.data_editor(
            reviews_df.drop(columns=["Year"]), 
            column_config={"Image 1": st.column_config.ImageColumn("Image 1"), 
                            "Image 2": st.column_config.ImageColumn("Image 2")},
            hide_index=True)

    ## RUN MODEL TAB 1
    reviews_df["Review"] = reviews_df["Review"].apply(clean_text)

    if run_model1:
        with st.spinner('Wait for it...'):
            sentiment_analyzer = create_analyzer(task="sentiment", lang="en")
            list_reviews = reviews_df["Review"].to_list()
            predictions = []
            positive = []
            negative = []
            neutral = []
        
            for review in list_reviews:
                #if review.split(" ")
                q = sentiment_analyzer.predict(review)

                predictions.append(q.output)
                positive.append(q.probas["POS"])
                negative.append(q.probas["NEG"])
                neutral.append(q.probas["NEU"])

        # Results
        df_results = reviews_df.copy()
        df_results["Result"] = predictions
        df_results["Result"] = df_results["Result"].map({"NEU":"Neutral", "NEG":"Negative", "POS":"Positive"})
        df_results["Negative"] = np.round(np.array(negative)*100)
        df_results["Neutral"] = np.round(np.array(neutral)*100)
        df_results["Positive"] = np.round(np.array(positive)*100)
        
        st.markdown("  ")

        tab1, tab2, tab3 = st.tabs(["All results", "Results per state", "Results per year"])        
        
        with tab1: # Overall results (tab_1)
            # get results df
            df_results_tab1 = df_results[["ID","Review","Rating","Negative","Neutral","Positive","Result"]]

            # warning message
            df_warning = df_results_tab1["Result"].value_counts().to_frame().reset_index()
            df_warning["Percentage"] = (100*df_warning["count"]/df_warning["count"].sum()).round(2)
            
            perct_negative = df_warning.loc[df_warning["Result"]=="Negative","Percentage"].to_numpy()[0]
            if perct_negative > 50:
                st.error(f"**Negative reviews alert** ⚠️: The proportion of negative reviews is {perct_negative}% !")

            # show dataframe results
            st.data_editor(
                df_results_tab1, #.loc[df_results_tab1["Customer ID"].isin(filter_customers)],
                column_config={
                    "Negative": st.column_config.ProgressColumn(
                        "Negative 👎",
                        help="Negative score of the review",
                        format="%d%%",
                        min_value=0,
                        max_value=100),
                    "Neutral": st.column_config.ProgressColumn(
                        "Neutral ✋",
                        help="Neutral score of the review",
                        format="%d%%",
                        min_value=0,
                        max_value=100),
                    "Positive": st.column_config.ProgressColumn(
                        "Positive 👍",
                        help="Positive score of the review",
                        format="%d%%",
                        min_value=0,
                        max_value=100)},
                    hide_index=True,
            )

        with tab2: # Results by state (tab_1)
            avg_state = df_results[["State","Negative","Neutral","Positive"]].groupby(["State"]).mean().round()
            avg_state = avg_state.reset_index().melt(id_vars="State", var_name="Sentiment", value_name="Score (%)")

            chart_state = alt.Chart(avg_state).mark_bar().encode(
                x=alt.X('Sentiment', axis=alt.Axis(title=None, labels=False, ticks=False)),
                y=alt.Y('Score (%)', axis=alt.Axis(grid=False)),
                color='Sentiment',
                column=alt.Column('State', header=alt.Header(title=None, labelOrient='bottom'))
            ).configure_view(
                stroke='transparent'
            ).interactive()

            st.altair_chart(chart_state)

            st.markdown("**Note**: The sentiment scores were computed on a small number of reviews.")


        with tab3: # Results by year (tab_1)
            avg_year = df_results[["Year","Negative","Neutral","Positive"]].groupby(["Year"]).mean().round()
            avg_year = avg_year.reset_index().melt(id_vars="Year", var_name="Sentiment", value_name="Score (%)")

            chart_year = alt.Chart(avg_year).mark_line().encode(
                x='Year',
                y='Score (%)',
                color='Sentiment',
            ).interactive()

            st.altair_chart(chart_year, use_container_width=True)


with tab2_:
    st.markdown("**Write your own review**")

    txt_review = st.text_area(
        "Write your review",
        "I recently visited a local Starbucks, and unfortunately, my experience was far from satisfactory. "
        "From the moment I stepped in, the atmosphere felt chaotic and disorganized. "
        "The staff appeared overwhelmed, leading to a significant delay in receiving my order. "
        "The quality of my drink further added to my disappointment. " 
        "The coffee tasted burnt, as if it had been sitting on the burner for far too long.",
        label_visibility="collapsed"
        )
    
    run_model2 = st.button("**Run the model**", type="primary", key="tab2")
        
    if run_model2:
        with st.spinner('Wait for it...'):
            sentiment_analyzer = create_analyzer(task="sentiment", lang="en")
            q = sentiment_analyzer.predict(txt_review)

            df_review_user = pd.DataFrame({"Polarity":["Positive","Neutral","Negative"], 
                        "Score":[q.probas['POS'], q.probas['NEU'], q.probas['NEG']]})

            fig = px.bar(df_review_user, x='Score', y='Polarity', color="Polarity", title='Polarity results', orientation="h")
            st.plotly_chart(fig, use_container_width=True)

        