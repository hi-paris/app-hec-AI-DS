import os
import altair as alt
import streamlit as st
import pandas as pd
import numpy as np
import plotly
import plotly.express as px


st.set_page_config(layout="wide")

@st.cache_data
def load_data_pickle(path,file):
    """ Load data from pickle file"""
    df = pd.read_pickle(os.path.join(path,file))
    return df


st.markdown("# Supervised vs Unsupervised Learning")

st.info("""There are two main types of models in the field of Data Science, **Supervised** and **Unsupervised learning** models. 
        Being able to distinguish which type of model fits your data is an essential step in building any AI project.""")

# st.markdown("""There are two main types of models in the field of Data Science, **Supervised** and **Unsupervised learning** models. <br> Being able to distinguish which type of model fits your application is essential step in building an AI project.
# - In **Supervised Learning**, models are trained by learning from "labeled data", which is data that contains the expected value to predict. 
#             Labeled data provides to the model the desired output, which it will then use to learn patterns and make predictions.   
# - In **Unsupervised Learning**, models learn the data's inherent structure without any specific guidance or instruction. The algorithm will identify any naturally occurring patterns in the dataset using "unlabeled data".
# """, unsafe_allow_html=True)

st.markdown(" ")
st.markdown("## What are the differences between both ?")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("#### Supervised Learning")
    st.markdown("""Models are trained by learning from "labeled data", which is data that contains the expected value to predict. 
            Labeled data provides to the model the desired output, which it will then use to learn patterns and make predictions.""")
    st.image("images/supervised_learner.png", caption="Supervised Learning")

with col2:
    st.markdown("#### Unsupervised Learning")
    st.markdown("""Models learn the data's inherent structure without any specific guidance or instruction. 
                The algorithm will identify any naturally occurring patterns in the dataset using "unlabeled data". """)
    st.image("images/unsupervised_learner.webp", caption="Unsupervised Learning")

st.divider()

st.markdown("## Try it out yourself")

# learning_type = st.radio(
#     "Select a type of Learning",
#     ["**Supervised Learning**", "**Unsupervised Learning**"],
#     captions = ["Make prediction based on labeled data", "Discover patterns in the data  without supervision"])

learning_type = st.selectbox("**Select a type of model**", 
                             ["Supervised Learning", 
                           "Unsupervised Learning"])



#######################################################################################################################
#                                           SUPERVISED LEARNING
#######################################################################################################################


if learning_type == "Supervised Learning":
    sl_usecase = st.selectbox("**Choose a use case**", 
                          ["Credit score classification 💳", 
                           "Customer churn prediction ❌"])
    
    st.divider()
    
    path_data_supervised = r"C:\Users\LaurèneDAVID\Documents\Teaching\Educational_apps\app-hec-AI-DS\data\classification"
    
    ################################# CREDIT SCORE ######################################
    
    if sl_usecase == "Credit score classification 💳":
        #st.divider()
        st.markdown("## Use case: Credit score classification 💳")
        st.info("""**Classification** is a type of supervised learning where the goal is to categorize input data into predefined classes or categories. 
                In this case, we will build a **credit score classification** model which will be able to predict if a client will have a **'Bad'**, **'Standard'** or **'Good'**.""")
        
        st.markdown("""**About the data:** <br> 
                    To train the credit classification model, you were provided bank and credit-related data of around 7600 clients. <br> 
                    This dataset is 'labeled' since it contains information on what we are trying to predict, which is the **Credit_Score** variable.""", 
                    unsafe_allow_html=True)
        
        credit_train = load_data_pickle(path_data_supervised, "credit_score_train.pkl")
        st.dataframe(credit_train.head())



    ################################# CUSTOMER CHURN #####################################
        
    elif sl_usecase == "Customer churn prediction ❌":
        pass







#######################################################################################################################
#                                           UNSUPERVISED LEARNING
#######################################################################################################################


def markdown_general_info(df):
    text = st.markdown(f"""
- **Age**: {int(np.round(df.Age))}
- **Yearly income**: {int(df.Income)} $
- **Number of kids**: {df.Kids}
- **Days of enrollment**: {int(np.round(df.Days_subscription))}
- **Web visits per month**: {df.WebVisitsMonth}    
""")   
    return text


if learning_type == "Unsupervised Learning":
    usl_usecase = st.selectbox("**Choose a use case**", 
                          ["Customer segmentation 🧑‍🤝‍🧑"])
    

    #################################### CUSTOMER SEGMENTATION ##################################

    path_clustering = r"data/clustering"
    path_clustering_results = r"data/clustering/results"

    if usl_usecase == "Customer segmentation 🧑‍🤝‍🧑":

        st.divider()
        st.markdown("### Customer Segmentation (Clustering) 🧑‍🤝‍🧑 ")

        st.info("""**Unsupervised learning** methods, such as clustering, are valulable tools for cases where you want a model to discover patterns by itself, without having to give it examples to learn from.
                    They can be useful for companies that want to perform **Customer Segmentation**. 
                    The AI clustering model can identify unknown groups of clients, which in turn helps the company create more targeted add campaigns, based on their consumer's behavior and preferences.
        """)

        st.markdown("  ")
        _, col, _ = st.columns([0.2,0.5,0.3])
        with col:
            st.image("images/cs.webp")

        st.markdown("### About the use case 📋")
        st.markdown("""You are giving a database that contains information on around 2000 customers of a mass-market retailer. 
                    The database's contains personal information (age, income, number of kids...), as well as information on the client's behavior. 
                    This includes what types of products were purchased by the client, how long has he been enrolled as a client and where these purchases were made. """, unsafe_allow_html=True)

        see_data = st.checkbox('**See the data**', key="dataframe")

        if see_data:
            customer_data = load_data_pickle(path_clustering, "clean_marketing.pkl") 
            st.dataframe(customer_data)

        learn_data = st.checkbox('**Learn more about the variables**', key="variable")

        if learn_data:
            st.markdown("""
        - **Age**: Customer's age
        - **Income**: Customer's yearly household income
        - **Kids**: Number of children/teenagers in customer's household
        - **Days_subscription**: Number of days since a customer's enrollment with the company
        - **Recency**: Number of days since customer's last purchase
        - **Wines**: Proportion of money spent on wine in last 2 years
        - **Fruits**: Proportion of money spent on fruits in last 2 years
        - **MeatProducts**: Proportion of money spent on meat in last 2 years
        - **FishProducts**: Proportion of money spent on fish in last 2 years
        - **SweetProducts**: Proportion of money spent sweets in last 2 years
        - **DealsPurchases**: Propotition of purchases made with a discount
        - **WebPurchases**: Propotition of purchases made through the company’s website
        - **CatalogPurchases**: Propotition of purchases made using a catalogue
        - **StorePurchases**: Propotition of purchases made directly in stores
        - **WebVisitsMonth**: Propotition of visits to company’s website in the last month""")
            st.divider()


        st.markdown(" ")
        st.markdown(" ")

        st.markdown("#### Clustering algorithm ⚙️")

        st.info("""**Clustering** is a type of unsupervised learning method that learns how to group similar data points together into "clusters", without needing supervision. 
                    In our case, a data points represents a customer that will be assigned to an unknown group.""")
        
        st.markdown(""" 
- The clustering algorithm used in this use case allows a specific number of groups to be identified, which isn't the case for all clustering models.
- The number of clusters chosen by the user can have a strong impact on the quality of the segmentation. Try to run the model multiple times with different number of clusters and see which number leads to groups with more distinct customer behaviors/preferences.""")
        st.markdown(" ")
        st.markdown("Here is an example of grouped data using a clustering model.")
        st.image("images/clustering.webp")


        nb_groups = st.selectbox("Choose a number of customer groups to identify", np.arange(2,6))
        df_results = load_data_pickle(path_clustering_results, f"results_{nb_groups}_clusters.pkl")

        st.markdown("  ")
        run_model = st.button("**Run the model**")
        #tab1, tab2 = st.tabs(["Results per product type", "Results per channel"])
        #st.divider()

        if run_model:
            cols_group = st.columns(int(nb_groups))
            for nb in range(nb_groups):
                df_nb = df_results[nb]

                col1, col2 = st.columns([0.3,0.7])
                with col1:
                    st.image("images/group.png", width=200)
                    st.header(f"Group {nb+1}", divider="grey")
                    markdown_general_info(df_nb)

                with col2:
                    tab1, tab2 = st.tabs(["Results per product type", "Results per channel"])
                    list_product_col = [col for col in list(df_nb.index) if "Products" in col]
                    df_products = df_nb.reset_index()
                    df_products = df_products.loc[df_products["variable"].isin(list_product_col)]
                    df_products.columns = ["variables", "values"]
                    
                    with tab1:
                        fig = px.pie(df_products, values='values', names='variables', 
                                        title="Amount spent per product type (in %)")
                        st.plotly_chart(fig, width=300)

                    list_purchases_col = [col for col in list(df_nb.index) if "Purchases" in col]
                    df_products = df_nb.reset_index()
                    df_products = df_products.loc[df_products["variable"].isin(list_purchases_col)]
                    df_products.columns = ["variables", "values"]
                    
                    with tab2:
                        fig = px.pie(df_products, values='values', names='variables', 
                                    title='Proportion of purchases made per channel (in %)')
                        st.plotly_chart(fig, width=300)