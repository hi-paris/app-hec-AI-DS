import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import numpy as np

from PIL import Image
from prophet import Prophet
from datetime import date
from utils import load_data_pickle
from sklearn.metrics import root_mean_squared_error


st.set_page_config(layout="wide")

@st.cache_data(ttl=3600, show_spinner=False)
def forecast_prophet(train, test, col=None):
    model = Prophet(daily_seasonality=False)
    
    for col in select_add_var:
        model.add_regressor(col)
    
    model.fit(train)
    forecast = model.predict(test)
    return model, forecast


###################################### TITLE ####################################

#st.image("images/ts_header.png")
st.markdown("# Time Series Forecasting")

st.markdown("### What is Time Series Forecasting ?")
st.info("""Time series forecasting models are AI models built to make accurate predictions about future values using historical data. 
            These types of models take into account temporal patterns, such as **trends** (long-term movements), **seasonality** (repeating patterns at fixed intervals), and **cyclic patterns** (repeating patterns not necessarily at fixed intervals)""")
            #unsafe_allow_html=True)

st.markdown(" ")
image_ts = Image.open('images/ts_patterns.png')
_, col, _ = st.columns([0.15,0.7,0.15])
with col:
    st.image(image_ts)

st.markdown("    ")

st.markdown("""Real-life applications of time series forecasting include:
- **Finance 💰**: Predict stock prices based on historical data to assist investors and traders in making informed decisions.
- **Energy ⚡**: Forecast energy consumption patterns to optimize resource allocation, plan maintenance, and manage energy grids more efficiently.
- **Retail 🏬**: Predict future demand for products to optimize inventory levels, reduce holding costs, and improve supply chain efficiency.
- **Transportation and Traffic flow :car:**: Forecasting traffic patterns to optimize route planning, reduce congestion, and improve overall transportation efficiency.
- **Healthcare** 👨‍⚕️: Predicting the number of patient admissions to hospitals, helping healthcare providers allocate resources effectively and manage staffing levels.
- **Weather 🌦️**: Predicting weather conditions over time, which is crucial for planning various activities, agricultural decisions, and disaster preparedness.
""")



st.markdown("    ")




###################################### USE CASE #######################################

# LOAD DATASET
path_timeseries = r"data/household"
data_model = load_data_pickle(path_timeseries,"household_power_consumption_clean.pkl")
data_model.rename({"Date":"ds", "Global_active_power":"y"}, axis=1, inplace=True)
data_model.dropna(inplace=True)
data_model["ds"] = pd.to_datetime(data_model["ds"])

# BEGINNING OF USE CASE
st.divider()
st.markdown("### Power Consumption Forecasting ⚡")

#st.markdown("  ")
st.info("""In this use case, a time series forecasting model is used to predict the **energy consumption** (or **Global Active Power**) of a household using historical data. 
        A forecasting model can be a valuable tool to optimize resource planning and avoid overloads during peak demand periods.""")

st.markdown(" ")

_, col, _ = st.columns([0.15,0.7,0.15])
with col:
    st.image("images/energy_consumption.jpg")

st.markdown("    ")
st.markdown("    ")

st.markdown("#### About the data 📋")

st.markdown("""You were provided data from the **daily energy consumption** of a household between January 2007 and November 2010 (46 months). <br>
            The goal is to forecast the **Global active power** being produced daily by the household. 
            Additional variables such as *Global Intensity* and three levels of *Sub-metering* are also available for the forecast.
            """, unsafe_allow_html=True)

st.markdown(" ")

st.info("""The data has been split into "historical data" and "to be predicted". Since forecasting models are **supervised**, we will use the household's energy data from January 2007 to December 2009 as historical data to train the model.
        We will then use the rest of the available data (starting January 2010) to test the performance of the model.""")

select_cutoff_date = date(2010, 1, 1)
select_cutoff_date = select_cutoff_date.strftime('%Y-%m-%d')

# SELECT TRAIN/TEST SET
train = data_model[data_model["ds"] <= select_cutoff_date]
test = data_model[data_model["ds"] > select_cutoff_date]

# PLOT TRAIN/TEST SET
train_plot = train.copy()
train_plot["split"] = ["historical data"]*len(train_plot)

test_plot = test.copy()
test_plot["split"] = ["to be predicted"]*len(test_plot)
data_clean_plot = pd.concat([train_plot, test_plot]) # plot dataset

st.markdown(" ")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Global active power", "Sub metering 1", "Sub metering 2", "Sub metering 3", "Global Intensity"])

with tab1:
    ts_chart = alt.Chart(data_clean_plot).mark_line().encode(
        x=alt.X('ds:T', axis=alt.Axis(format='%b %Y', tickCount=12), title="Date"),
        y=alt.Y('y:Q', title="Global active power"),
        color='split:N',
    ).interactive()

    st.markdown("**View Global active power** (to be forecasted)")
    st.altair_chart(ts_chart, use_container_width=True)
    st.success("""**Global active power** refers to the total real power consumed by electrical devices in the house (in kilowatts).""")
    

with tab2:
    ts_chart = alt.Chart(data_clean_plot.loc[data_clean_plot["split"]=="historical data"]).mark_line().encode(
                x=alt.X('ds:T', axis=alt.Axis(format='%b %Y', tickCount=12), title="Date"),
                y=alt.Y('Sub_metering_1:Q', title="Sub metering 1"),
                color=alt.Color('split:N')) #, scale=custom_color_scale))
            
    st.markdown("**View Sub-metering 1** (additional)")
    st.altair_chart(ts_chart.interactive(), use_container_width=True)
    st.success("**Sub-metering 1** is the total active power consumed by the kitchen in the house (in kilowatts).")



with tab3:    
    ts_chart = alt.Chart(data_clean_plot.loc[data_clean_plot["split"]=="historical data"]).mark_line().encode(
        x=alt.X('ds:T', axis=alt.Axis(format='%b %Y', tickCount=12), title="Date"),
        y=alt.Y('Sub_metering_2:Q', title="Sub metering 2"),
        color=alt.Color('split:N')) #, scale=custom_color_scale))

    st.markdown("**View Sub-metering 2** (additional)")
    st.altair_chart(ts_chart.interactive(), use_container_width=True)
    st.success("**Sub-metering 2** is the total active power consumed by the laundry room in the house (in kilowatts).")


with tab4:        
    ts_chart = alt.Chart(data_clean_plot.loc[data_clean_plot["split"]=="historical data"]).mark_line().encode(
            x=alt.X('ds:T', axis=alt.Axis(format='%b %Y', tickCount=12), title="Date"),
            y=alt.Y('Sub_metering_3:Q', title="Sub metering 3"),
            color=alt.Color('split:N')) #scale=custom_color_scale))

    st.markdown("**View Sub-metering 3** (additional)")
    st.altair_chart(ts_chart.interactive(), use_container_width=True)
    st.success("**Sub-metering 3** is the active power consumed by the electric water heater and air conditioner in the household (in kilowatts).")


with tab5:
    custom_color_scale = alt.Scale(range=['red', 'lightcoral'])
    ts_chart = alt.Chart(data_clean_plot.loc[data_clean_plot["split"]=="historical data"]).mark_line().encode(
        x=alt.X('ds:T', axis=alt.Axis(format='%b %Y', tickCount=12), title="Date"),
        y=alt.Y('Global_intensity:Q', title="Global active power"),
        color=alt.Color('split:N')) # scale=custom_color_scale))

    st.markdown("**View Global intensity** (additional)")
    st.altair_chart(ts_chart.interactive(), use_container_width=True)
    st.success("**Global intensity** is the average current intensity delivered to the household (amps).")



st.markdown(" ")
st.markdown(" ")
st.markdown("#### Forecast model 📈")
st.markdown("""The forecasting model used in this use case allows **additional data** to be used for training. 
        Try adding more data to the model as it can help improve its performance and accuracy.""")



# ADD VARIABLES TO ANALYSIS
add_var = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3","Global_intensity"]
st.markdown("")
select_add_var = st.multiselect("**Add variables to the model**", add_var)

if 'model_train' not in st.session_state:
    st.session_state['model_train'] = False
    
# if st.session_state.model_train:
#     text = "The model has alerady been trained."
# else:
#     st.write("The model hasn't been trained yet")

st.markdown("")
run_model = st.button("**Run the model**")


st.markdown("    ")
st.markdown("    ")





################################## SEE RESULTS ###############################

if "saved_model" not in st.session_state:
    st.session_state["saved_model"] = False


if run_model:
    with st.spinner('Wait for it...'):
        fbmodel, forecast = forecast_prophet(train, test, col=select_add_var)
        st.session_state.model_train = True
        st.session_state.saved_model = fbmodel

        ####################### SEE RESULTS ########################
        st.markdown("#### See the results ☑️")
        st.info("The model is able to forecast energy consumption as well as learn the predicted data's **trend**, **weekly** and **yearly seasonality**.")

        tab1_result, tab2_result, tab3_result, tab4_result = st.tabs(["Performance", "Trend", "Weekly seasonality", "Yearly seasonality"])
        with tab1_result:
            # Compute model root mean squared error
            y_true = test_plot["y"]
            y_pred = forecast["yhat"]
            error = str(np.round(root_mean_squared_error(y_true, y_pred, ),3))

            col1, col2 = st.columns([0.1,0.9])

            with col1:
                st.markdown("")
                st.metric(label="**Average error**", value=error)
            
            with col2:
                # Create df for true vs predicted plot
                df_results = pd.concat([test_plot.reset_index(drop=True), forecast.drop(columns=["ds"]).reset_index(drop=True)], axis=1)[["ds","y","yhat"]]
                df_results = df_results.melt(id_vars="ds")
                df_results["variable"] = df_results["variable"].map({"y":"true values", "yhat":"predicted values"})
                df_results.columns = ["Date", "Variable", "Global Active Power"]
                
                fig = px.line(df_results, x="Date", y="Global Active Power", color="Variable", 
                        color_discrete_sequence=["lightblue", "black"], line_dash = 'Variable')

                fig.update_layout(
                    title=f'True vs predicted power consumption',
                    width=1200,  
                    height=600  
                )

                st.plotly_chart(fig, use_container_width=True)

        with tab2_result:
            ymin = forecast["trend"].min() 
            ymax = forecast["trend"].max()
            
            fig = px.area(forecast, x="ds", y="trend", color_discrete_sequence=["red"]) #range_y=[ymin, ymax])
            fig.update_layout(title="Trend", xaxis_title="Date", yaxis_title="Trend")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""**Interpretation** <br>
                        No trend in the household's energy consumption has been detected by the model.""", unsafe_allow_html=True)

        with tab3_result:
            #st.success("**Weekly seasonality** refers to a repeating pattern or variation that occurs on a weekly basis on the energy consumption data.")
            days_week = dict(zip(np.arange(1,8),["Monday", "Tuedsay", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]))
            forecast_weekly = forecast.copy()
            forecast_weekly["dayweek"] = forecast_weekly["ds"].apply(lambda x: x.isoweekday()).map(days_week)

            fig = px.area(forecast_weekly, x="dayweek", y="weekly", color_discrete_sequence=["purple"])
            fig.update_layout(title="Weekly seasonality", xaxis_title="Date", yaxis_title="Weekly")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""**Interpretation** <br>
            The household consumes more electrical power during the week-end (Saturday and Sunday) then during the week.           
                        """, unsafe_allow_html=True)

        with tab4_result:
            forecast_year = forecast[["ds","yearly"]].copy()
            forecast_year["ds_year"] = forecast_year["ds"].apply(lambda x: x.strftime("%B %d"))
            forecast_year["ds"] = forecast_year["ds"].apply(lambda x: x.strftime("%m-%d"))
            forecast_year.sort_values(by=["ds"], inplace=True)
            forecast_year = forecast_year.groupby(["ds","ds_year"]).mean().reset_index()

            st.markdown("")
            ts_chart = alt.Chart(forecast_year, title="Yearly seasonality").mark_area(opacity=0.5,line = {'color':'darkblue'}).encode(
                x=alt.X('ds_year:T', axis=alt.Axis(format='%b', tickCount=12), title="Date"),
                y=alt.Y('yearly:Q', title="Yearly seasonality"),
                ).interactive()

            st.altair_chart(ts_chart, use_container_width=True)

            st.markdown("""**Interpretation** <br>
            The household consumes more energy during the winter (November to February) and less during the warmer months.           
                        """, unsafe_allow_html=True)
            


################################## MAKE FUTURE PREDICTIONS ###############################

# st.markdown("#### Forecast new values ")

# st.info("**The model needs to be trained before it can predict new values.**")

# st.

# make_predictions = st.button("**Forecast new values**")

# if make_predictions is True:
#     if st.session_state.saved_model is True:


