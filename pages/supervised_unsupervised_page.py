import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils import load_data_pickle, load_model_pickle
from annotated_text import annotated_text

#####################################################################################
#                                    PAGE CONFIG
#####################################################################################

st.set_page_config(layout="wide")



#####################################################################################
#                                    INTRO 
#####################################################################################


st.markdown("# Supervised vs Unsupervised Learning")

st.info("""There are two main types of models in the field of Data Science, **Supervised** and **Unsupervised learning** models. 
        Being able to distinguish which type of model fits your data is an essential step in building any AI project.""")

st.markdown(" ")
#st.markdown("## What are the differences between both ?")

col1, col2 = st.columns(2, gap="large")

with col1: 
    st.markdown("### Supervised Learning")
    st.markdown("""In supervised learning, models are trained by learning from **labeled data**. <br>
                Labeled data provides to the model the desired output, which it will then use to learn relevant patterns and make predictions. 
- A model is first **trained** to make predictions using labeled data
- The trained model can then be used to **predict values** for new data.
                """, unsafe_allow_html=True)
    st.markdown(" ")
    st.image("images/supervised_learner.png", caption="An example of supervised learning")

with col2:
    st.markdown("### Unsupervised Learning")
    st.markdown("""In unsupervised learning, models **learn the data's inherent structure** without any explicit guidance on what to look for.
                The algorithm will identify any naturally occurring patterns in the dataset using **unlabeled data**. 
- They can be useful for applications where the goal is to discover **unknown groupings** in the data.
- They are also used to identify unusual patterns or **outliers**.
                """, unsafe_allow_html=True)
    st.markdown(" ")
    st.image("images/unsupervised_learner.webp", caption="An example of unsupervised Learning")

st.markdown("  ")

learning_type = st.selectbox("**Select a type of model**", 
                             ["Supervised Learning", 
                           "Unsupervised Learning"])





#######################################################################################################################
#                                               SUPERVISED LEARNING
#######################################################################################################################


if learning_type == "Supervised Learning":
    sl_usecase = st.selectbox("**Choose a use case**", 
                          ["Credit score classification 💯", 
                           "Customer churn prediction ❌"])
    
    st.markdown(" ")
    
    # st.divider()
    
    path_data_supervised = r"data/classification"
    path_pretrained_supervised = r"pretrained_models/supervised_learning"
    
    ################################# CREDIT SCORE ######################################
    
    if sl_usecase == "Credit score classification 💯":

        path_credit = os.path.join(path_data_supervised,"credit_score")
        
        ## Description of the use case
        st.divider()
        st.markdown("## Credit score classification 💯")
        st.info("""**Classification** is a type of supervised learning where the goal is to categorize input data into predefined classes or categories. 
                In this case, we will build a **credit score classification** model that predicts if a client will have a **'Bad'**, **'Standard'** or **'Good'** credit score.""")
        st.markdown(" ")

        _, col, _ = st.columns([0.25,0.5,0.25])
        with col:
            st.image("images/credit_score.jpg")

        ## Learn about the data
        st.markdown("#### About the data 📋")
        st.markdown("""To train the credit classification model, you were provided a **labeled** database with the bank and credit-related information of around 7600 clients. <br> 
                    This dataset is 'labeled' since it contains information on what we are trying to predict, which is the **Credit_Score** variable.""", 
                    unsafe_allow_html=True)
        
        ## Load data 
        credit_train = load_data_pickle(path_credit, "credit_score_train_raw.pkl")
        credit_test_pp = load_data_pickle(path_credit, "credit_score_test_pp.pkl")
        labels = ["Good","Poor","Standard"]

        ## Load model
        credit_model = load_model_pickle(path_pretrained_supervised,"credit_score_model.pkl")

        # View data
        see_data = st.checkbox('**See the data**', key="credit_score\data")
        if see_data:
            st.warning("The data of only the first 30 clients are shown.") 
            st.dataframe(credit_train.head(30).reset_index(drop=True))

        learn_data = st.checkbox('**Learn more about the data**', key="credit_score_var")
        if learn_data:
            st.markdown("""
- **Age**: The client's age
- **Occupation**: The client's occupation/job
- **Credit_Mix**: Score for the different type of credit accounts a client has (mortgages, loans, credit cards, ...)
- **Payment_of_Min_Amount**: Whether the client is making the minimum required payments on their credit accounts (Yes, No, NM:Not mentioned)
- **Annual_Income**: The client's annual income
- **Num_Bank_Accounts**: Number of bank accounts opened 
- **Num_Credit_Card**: Number of credit cards owned
- **Interest_Rate**: The client's average interest rate 
- **Num_of_Loan**: Number of loans of the client
- **Changed_Credit_Limit**: Whether a client changed his credit limit once or not (Yes, No)                 - 
- **Outstanding Debt**: A client's outstanding debt
- **Credit_History_Age**: The length of a client's credit history (in months)
""")
            
        st.markdown("  ")
        st.markdown("  ")

        ## Train the algorithm
        st.markdown("#### Train the algorithm ⚙️")
        st.info("""**Training** an AI model means feeding it data that contains multiple examples of clients with their credit scores. 
                Using the labeled data provided, the model will **learn relationships** between a client's credit score and the other bank/credit-related variables provided.
                Using these learned relationships, the model will then try to make **accurate predictions**.""")

        # st.markdown("""Before feeding the model data for training, exploratory data analysis is often conducted to discover if patterns can discovered beforehand.""")            
        # st.image("images/models/credit_score/EDA_numeric_credit.png")
        #st.markdown("In our case, the training data is the dataset containing the bank and credit information of our 7600 customers.")

        if 'model_train' not in st.session_state:
            st.session_state['model_train'] = False

        if st.session_state.model_train:
            st.write("The model has been trained.")
        else:
            st.write("The model hasn't been trained yet")

        run_credit_model = st.button("**Train the model**")

        if run_credit_model:
            st.session_state.model_train = True
            with st.spinner('Wait for it...'):
                st.markdown(" ")
                st.markdown(" ")
                time.sleep(2)
                st.markdown("#### See the results ☑️")
                tab1, tab2 = st.tabs(["Performance", "Explainability"])
                
                ######## MODEL PERFORMANCE 
                with tab1:
                    results_train = load_data_pickle(path_credit,"credit_score_cm_train")
                    results_train = results_train.to_numpy()
                    accuracy = np.round(results_train.diagonal()*100)
                    df_accuracy = pd.DataFrame({"Credit Score":["Good","Poor","Standard"],
                                                "Accuracy":accuracy})

                    st.markdown(" ")
                    st.info("""**Evaluating a model's performance** helps provide a quantitative measure of the model's ability to make accurate decisions.
                                In this use case, the performance of the credit score model was measured by comparing clients' true credit scores with the scores predicted by the trained model.""")

                    fig = px.bar(df_accuracy, y='Accuracy', x='Credit Score', color="Credit Score", title="Model performance")
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("""<i>The model's accuracy was measured for every type of credit score (Good, Standard, Poor).</i>
                                <i>This is crucial as to understand whether the model is consistant in its performance, or whether it has trouble distinguishing between two kinds of credit score.</i>""", 
                                unsafe_allow_html=True)

                    st.markdown(" ")

                    st.markdown("""**Interpretation**: <br>
                                Our model's is overall quite accurate in predicting all types of credit scores with an accuracy that is above 85% for each. 
                                We do note that is slighly more accuracte in predicting a good credit score (92%) and less for a standard credit score (86%). 
                                This can be due to the model having a harder time distinguishing between clients with a standard credit score and other more "extreme" credit scores (Good, Bad).
                                """, unsafe_allow_html=True)
                    
                ##### MODEL EXPLAINABILITY
                with tab2:
                    st.markdown(" ")
                    st.info("""**Explainability** in AI refers to the ability to understand which variable used by a model during training had the most impact on the final predictions and how to quantify this impact.
                            Understanding the inner workings of a model helps build trust among users and stakeholders, as well as increase acceptance.""")
                    
                    # Create feature importance dataframe
                    df_var_importance = pd.DataFrame({"variable":credit_test_pp.columns,
                                                    "score":credit_model.feature_importances_})
                    
                    # Compute average score for categorical variables
                    for column in ["Occupation","Credit_Mix","Payment_of_Min_Amount"]:
                        col_remove = [col for col in credit_test_pp.columns if f"{column}_" in col]
                        avg_score = df_var_importance.loc[df_var_importance["variable"].isin(col_remove)]["score"].mean()
                        
                        df_var_importance = df_var_importance.loc[~df_var_importance["variable"].isin(col_remove)]                    
                        new_row = pd.DataFrame([[column, avg_score]], columns=["variable","score"])
                        df_var_importance = pd.concat([df_var_importance, new_row], ignore_index=True)
                    
                    df_var_importance.sort_values(by=["score"], inplace=True)
                    df_var_importance["score"] = df_var_importance["score"].round(3)                

                    # Feature importance plot with plotly
                    fig = px.bar(df_var_importance, x='score', y='variable', color="score", orientation="h", title="Model explainability")
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("""<b>Interpretation</b>: <br>
                                A client's outstanding debt, interest rate and delay from due date were the most crucial factors in explaining their final credit score. <br>
                                Whether a client is making their minimum required payments on their credit accounts (Payment_min_amount), their occupation and their number of loans had a very limited impact on their credit score.,
                                """, unsafe_allow_html=True)
                    

        st.markdown(" ")
        st.markdown(" ")
        st.markdown("#### Predict credit score 🆕")
        st.info("You can only predict the credit score of new clients once the **model has been trained.**")
        st.markdown("  ")
        
        col1, col2 = st.columns([0.25,0.75], gap="medium")
        
        credit_test = load_data_pickle(path_credit,"credit_score_test_raw.pkl")
        credit_test.reset_index(drop=True, inplace=True)
        credit_test.insert(0, "Client ID", [f"{i}" for i in range(credit_test.shape[0])])
        #credit_test.drop(columns=["Credit_Score"], inplace=True)
        
        with col1:
            st.markdown("""<b>Filter the data</b> <br>
            You can select clients based on their *Age*, *Annual income* or *Oustanding Debt*.""", 
            unsafe_allow_html=True)

            select_image_box = st.radio(" ",
            ["Filter by Age", "Filter by Income", "Filter by Outstanding Debt", "No filters"],
            label_visibility="collapsed")

            if select_image_box == "Filter by Age":
                st.markdown(" ")
                min_age, max_age = st.slider('Select a range', credit_test["Age"].astype(int).min(), credit_test["Age"].astype(int).max(), (19,50), 
                                             key="age", label_visibility="collapsed")
                credit_test = credit_test.loc[credit_test["Age"].between(min_age,max_age)]

            if select_image_box == "Filter by Income":
                st.markdown(" ")
                min_income, max_income = st.slider('Select a range', credit_test["Annual_Income"].astype(int).min(), 180000, 
                                                   (7000, 100000), label_visibility="collapsed", key="income")
                credit_test = credit_test.loc[credit_test["Annual_Income"].between(min_income, max_income)]

            if select_image_box == "Filter by Outstanding Debt":
                min_debt, max_debt = st.slider('Select a range', credit_test["Outstanding_Debt"].astype(int).min(), credit_test["Outstanding_Debt"].astype(int).max(), 
                                                (0,2000), label_visibility="collapsed", key="debt")
                credit_test = credit_test.loc[credit_test["Outstanding_Debt"].between(min_debt, max_debt)]

            if select_image_box == "No filters":
                pass

        st.markdown(" ")
        st.markdown("""<b>Select a threshold for the alert</b> <br>
                    A warning message will be displayed if the percentage of poor credit scores exceeds this threshold.
                    """, unsafe_allow_html=True)
        warning_threshold = st.slider('Select a value', min_value=20, max_value=100, step=10, 
                                        label_visibility="collapsed", key="warning")

        st.markdown(" ")
        st.write("The threshold is at", warning_threshold, "%")
            

        with col2:
            #st.markdown("**View the database**")
            st.dataframe(credit_test)
        
        make_predictions = st.button("**Make predictions**")
        st.markdown(" ")

        if make_predictions:
            if st.session_state.model_train is True:
                X_test = credit_test_pp.iloc[credit_test.index,:]
                predictions = credit_model.predict(X_test)
                
                df_results_pred = credit_test.copy()
                df_results_pred["Credit Score"] = predictions
                df_mean_pred = df_results_pred["Credit Score"].value_counts().to_frame().reset_index()
                df_mean_pred.columns = ["Credit Score", "Proportion"]
                df_mean_pred["Proportion"] = (100*df_mean_pred["Proportion"]/df_results_pred.shape[0]).round()
                
                perct_bad_score = df_mean_pred.loc[df_mean_pred["Credit Score"]=="Poor"]["Proportion"].to_numpy()

                if perct_bad_score >= warning_threshold:
                    st.error(f"The proportion of clients with a bad credit score is above {warning_threshold}% (at {perct_bad_score[0]}%)⚠️")

                col1, col2 = st.columns([0.4,0.6], gap="large")
                with col1:
                    st.markdown("**Proporition of predicted credit scores**")
                    fig = px.pie(df_mean_pred, values='Proportion', names='Credit Score')
                                        #title="Proportion of credit scores")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    df_show_results = df_results_pred[["Credit Score","Client ID"] + [col for col in df_results_pred.columns if col not in ["Client ID","Credit Score"]]]                    
                    columns_float = df_show_results.select_dtypes(include="float").columns
                    df_show_results[columns_float] = df_show_results[columns_float].astype(int)
                    
                    def highlight_score(val):
                        if val == "Good":
                            color = 'red'
                        if val == 'Standard':
                            color= "cornflowerblue"
                        if val == "Poor":
                            color = 'blue'
                        return f'color: {color}'
                    
                    df_show_results_color = df_show_results.style.applymap(highlight_score, subset=['Credit Score'])
                    
                    st.markdown("**Overall results**")
                    st.dataframe(df_show_results_color)

            else:
                st.error("You have to train the credit score model first.")




    ################################# CUSTOMER CHURN #####################################
        
    elif sl_usecase == "Customer churn prediction ❌":
        st.warning("This page is under construction")





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

        # st.divider()
        st.divider()
        st.markdown("## Customer Segmentation (Clustering) 🧑‍🤝‍🧑")

        st.info("""**Unsupervised learning** methods, such as clustering, are valulable tools for cases where you want a model to discover patterns by itself, without having to give it examples to learn from.
                    They can be useful for companies that want to perform **Customer Segmentation**. 
                    The AI clustering model can identify unknown groups of clients, which in turn helps the company create more targeted add campaigns, based on their consumer's behavior and preferences.
        """)
        st.markdown("  ")

        ## Show image
        _, col, _ = st.columns([0.2,0.5,0.3])
        with col:
            st.image("images/cs.webp")

        ## About the use case
        st.markdown("#### About the use case 📋")
        st.markdown("""You are giving a database that contains information on around 2000 customers of a mass-market retailer. 
                    The database's contains personal information (age, income, number of kids...), as well as information on the client's behavior. 
                    This includes what types of products were purchased by the client, how long has he been enrolled as a client and where these purchases were made. """, unsafe_allow_html=True)

        see_data = st.checkbox('**See the data**', key="dataframe")

        if see_data:
            customer_data = load_data_pickle(path_clustering, "clean_marketing.pkl") 
            st.dataframe(customer_data.head(10))

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
        - **DealsPurchases**: Proportion of purchases made with a discount
        - **WebPurchases**: Proportion of purchases made through the company’s website
        - **CatalogPurchases**: Proporition of purchases made using a catalogue
        - **StorePurchases**: Proportion of purchases made directly in stores
        - **WebVisitsMonth**: Proportion of visits to company’s website in the last month""")
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
        
