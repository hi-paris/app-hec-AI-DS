import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import numpy as np

from PIL import Image
from prophet import Prophet
from datetime import date
from sklearn.metrics import root_mean_squared_error




st.set_page_config(layout="wide")


###################################### TITLE ####################################

#st.image("images/ts_header.png")
st.markdown("# Time Series Forecasting")

st.markdown("### What is Time Series Forecasting ?")
st.info("""Time series forecasting models are AI models built to make accurate predictions about future values using historical data. 
            These types of models take into account temporal patterns, such as **trends** (long-term movements), **seasonality** (repeating patterns at fixed intervals), and **cyclic patterns** (repeating patterns not necessarily at fixed intervals)""")
            #unsafe_allow_html=True)

st.markdown("The following image shows the difference patterns (trend, seasonality and cyclic) that can be identified by Time Series models.")
image_ts = Image.open('images/ts_patterns.png')
st.image(image_ts, width=1000)

st.markdown("    ")

st.markdown("""Here are some useful applications of time series forecasting:
- **Finance 💰**: Predict stock prices based on historical data to assist investors and traders in making informed decisions.
- **Energy ⚡**: Forecast energy consumption patterns to optimize resource allocation, plan maintenance, and manage energy grids more efficiently.
- **Retail 🏬**: Predict future demand for products to optimize inventory levels, reduce holding costs, and improve supply chain efficiency.
- **Transportation and Traffic flow :car:**: Forecasting traffic patterns to optimize route planning, reduce congestion, and improve overall transportation efficiency.
- **Healthcare** 👨‍⚕️: Predicting the number of patient admissions to hospitals, helping healthcare providers allocate resources effectively and manage staffing levels.
- **Weather 🌦️**: Predicting weather conditions over time, which is crucial for planning various activities, agricultural decisions, and disaster preparedness.
""")



st.markdown("    ")




###################################### USE CASE #######################################

st.divider()
st.markdown("### Power Consumption Forecasting ⚡")

#st.markdown("  ")
st.warning("""In this example, we built a time series forecasting model that predicts the **daily energy consumption of a household** (or *Global Active Power*) using historical data. 
            The data was gathered from a house in the South of France, between January 2007 and November 2010 (46 months).
            """)

st.image("images/power.jpg")

st.markdown("    ")

st.markdown("""**Information on the model**:
- Our model allows **additional data** (other than Global Active Power) to be given to the model to help improve its performance. Here, you can choose between *Global Intensity*, and the *three levels of Sub-metering* to help forecast the overall energy consumption of a household. 
- The data is split using a **cut-off date** between the historical data and the data we will predict. The historical data will allow the model to learn patterns in the data. The rest will allow us to evaluate the model's performance, by comparing the real and predicted values.
            """, unsafe_allow_html=True)


st.markdown("    ")
st.markdown("    ")


# LOAD DATASET
data = pd.read_csv(r"data/household/household_power_consumption_clean.csv")
data_model = data.copy()
data_model.rename({"Date":"ds", "Global_active_power":"y"}, axis=1, inplace=True)
data_model.dropna(inplace=True)
data_model["ds"] = pd.to_datetime(data_model["ds"])

col1, col2 = st.columns([0.2,0.8], gap="medium")

with col1:
    # SELECT CUTOFF DATE
    all_dates = data_model["ds"].sort_values().unique()
    #select_cutoff_date = '2010-01-01' # split train/test

# Create a date object for "2007-01-01"
    start_date = date(2009, 5, 1)
    end_date = date(2010, 9, 1)

    select_cutoff_date = st.slider(
    "**Select a cut-off date**",
    min_value=start_date,
    max_value=end_date,
    value=date(2010, 1, 1),
    format="YYYY-MM-DD")

    #st.write("Start time:", select_cutoff_date )

    #select_cutoff_date = st.date_input("**Select a cut-off date**", min_value=start_date, max_value=end_date, value=date(2010, 1, 1)) #datetime.date(2019, 7, 6))
    select_cutoff_date = select_cutoff_date.strftime('%Y-%m-%d')

    # ADD VARIABLES TO ANALYSIS
    add_var = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3","Global_intensity"]
    st.markdown("")
    select_add_var = st.multiselect("**Add variables to the model**", add_var)
    st.markdown("")
    run_model = st.button("**Run the model**", type="primary")

with col2:
    # SELECT TRAIN/TEST SET
    train = data_model[data_model["ds"] <= select_cutoff_date]
    test = data_model[data_model["ds"] > select_cutoff_date]

    # PLOT TRAIN/TEST SET
    train_plot = train.copy()
    train_plot["split"] = ["historical data"]*len(train_plot)

    test_plot = test.copy()
    test_plot["split"] = ["to be predicted"]*len(test_plot)
    data_clean_plot = pd.concat([train_plot, test_plot]) # plot dataset

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Global active power", "Sub metering 1", "Sub metering 2", "Sub metering 3", "Global Intensity"])

    with tab1:
        ts_chart = alt.Chart(data_clean_plot).mark_line().encode(
            x=alt.X('ds:T', title="Date"),
            y=alt.Y('y:Q', title="Global active power"),
            color='split:N',
        ).interactive()

        st.markdown("**View Global active power** (to be forecasted)")
        st.altair_chart(ts_chart, use_container_width=True)
        st.info("""**Global active power** refers to the total real power consumed by electrical devices in the house (in kilowatts).""")

    
    with tab2:
        ts_chart = alt.Chart(data_clean_plot).mark_line().encode(
                    x=alt.X('ds:T', title="Date"),
                    y=alt.Y('Sub_metering_1:Q', title="Sub metering 1"),
                    color=alt.Color('split:N')) #, scale=custom_color_scale))
                
        st.markdown("**View Sub-metering 1** (to be added)")
        st.altair_chart(ts_chart.interactive(), use_container_width=True)
        st.info("*Sub-metering 1* is the total active power consumed by the kitchen in the house (in kilowatts).")


    with tab3:
        ts_chart = alt.Chart(data_clean_plot).mark_line().encode(
            x=alt.X('ds:T', title="Date"),
            y=alt.Y('Sub_metering_2:Q', title="Sub metering 2"),
            color=alt.Color('split:N')) #, scale=custom_color_scale))

        st.markdown("**View Sub-metering 2** (to be added)")
        st.altair_chart(ts_chart.interactive(), use_container_width=True)
        st.info("*Sub-metering 2* is the total active power consumed by the laundry room in the house (in kilowatts).")

    with tab4:        
        ts_chart = alt.Chart(data_clean_plot).mark_line().encode(
                x=alt.X('ds:T', title="Date"),
                y=alt.Y('Sub_metering_3:Q', title="Sub metering 3"),
                color=alt.Color('split:N')) #scale=custom_color_scale))

        st.markdown("**View Sub-metering 3** (to be added)")
        st.altair_chart(ts_chart.interactive(), use_container_width=True)
        st.info("***Sub-metering 3* is the active power consumed by the electric water heater and air conditioner in the household (in kilowatts).")


    with tab5:
        custom_color_scale = alt.Scale(range=['red', 'lightcoral'])

        ts_chart = alt.Chart(data_clean_plot).mark_line().encode(
            x=alt.X('ds:T', title="Date"),
            y=alt.Y('Global_intensity:Q', title="Global active power"),
            color=alt.Color('split:N')) # scale=custom_color_scale))

        st.markdown("**View Global intensity** (to be added)")
        st.altair_chart(ts_chart.interactive(), use_container_width=True)
        st.info("**Global intensity** is the average current intensity delivered to the household (amps).")


# st.markdown(" ")



################################## PROPHET MODEL ###############################


if run_model:
    with st.spinner('Wait for it...'):
        # ADD REGRESSORS TO MODEL
        m = Prophet(daily_seasonality=False)
        for col in select_add_var:
            m.add_regressor(col)
        
        m.fit(train)
        forecast = m.predict(test)

        st.markdown("##### Results")

        tab1_result, tab2_result, tab3_result, tab4_result = st.tabs(["True vs Predicted", "Trend", "Weekly seasonality", "Yearly seasonality"])
        with tab1_result:
            # Compute model root mean squared error
            y_true = test_plot["y"]
            y_pred = forecast["yhat"]
            error = str(np.round(root_mean_squared_error(y_true, y_pred),3))

            # Create df for true vs predicted plot
            df_results = pd.concat([test_plot.reset_index(drop=True), forecast.drop(columns=["ds"]).reset_index(drop=True)], axis=1)[["ds","y","yhat"]]
            df_results = df_results.melt(id_vars="ds")
            df_results["variable"] = df_results["variable"].map({"y":"true values", "yhat":"predicted values"})
            df_results.columns = ["Date", "Variable", "Global Active Power"]
            
            fig = px.line(df_results, x="Date", y="Global Active Power", color="Variable", 
                    color_discrete_sequence=["blue", "black"], line_dash = 'Variable')

            fig.update_layout(
                title=f'True vs predicted power consumption (error={error})',
                width=1200,  
                height=600  
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab2_result:
            ymin = forecast["trend"].min() 
            ymax = forecast["trend"].max()
            
            fig = px.area(forecast, x="ds", y="trend", color_discrete_sequence=["red"], range_y=[ymin, ymax])
            fig.update_layout(title="Trend of the power consumption")
            st.plotly_chart(fig, use_container_width=True)

        with tab3_result:
            days_week = dict(zip(np.arange(1,8),["Sunday", "Monday", "Tuedsay", "Wednesday", "Thursday", "Friday", "Saturday"]))
            forecast_weekly = forecast.copy()
            forecast_weekly["dayweek"] = forecast_weekly["ds"].apply(lambda x: x.isoweekday()).map(days_week)

            fig = px.area(forecast_weekly, x="dayweek", y="weekly", color_discrete_sequence=["purple"])
            fig.update_layout(title="Weekly seasonality of the power consumption")
            st.plotly_chart(fig, use_container_width=True)

        with tab4_result:
            forecast_year = forecast[["ds","yearly"]].copy()
            forecast_year["ds_year"] = forecast_year["ds"].apply(lambda x: x.strftime("%B %d"))
            forecast_year["ds"] = forecast_year["ds"].apply(lambda x: x.strftime("%m-%d"))
            forecast_year.sort_values(by=["ds"], inplace=True)

            forecast_year = forecast_year.groupby(["ds","ds_year"]).mean().reset_index()
            fig = px.area(forecast_year, x="ds_year", y="yearly", color_discrete_sequence=["green"])
            fig.update_layout(title="Yearly seasonality of the power consumption")
            st.plotly_chart(fig, use_container_width=True)


#############################