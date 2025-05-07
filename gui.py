import streamlit as st
import pandas as pd
import requests
import time
import altair as alt
import matplotlib.pyplot as plt

import pandas as pd
# root_url = "https://honest-tiger-83.telebit.io"

root_url = "http://127.0.0.1:8000"

st.title("System Simulation Control")

height = []
time_data = []

# Sidebar for user inputs


system = st.sidebar.selectbox("Select System", ("ODE", "Real"))
setpoint = st.sidebar.number_input("Setpoint", min_value=0.0, max_value=10.0, value=0.1)
chart_container = st.empty()

# df = pd.DataFrame({'time':[],'height':[]})
# df.to_csv('cache.csv')
# Function to fetch and plot data
def fetch_and_plot_data():
    
    response_ode = requests.get(f"{root_url}/ODE/data")

    response_real = requests.get(f"{root_url}/ODE/data")
    
    data_real = response_real.json()
    data_ode = response_ode.json()
    new_real_ht = (float(data_real["height"]))
    new_real_time = (float(data_real["time"]))
    
    new_ode_ht = (float(data_real["height"]))
    new_ode_time = (float(data_real["time"]))
        
    new_rows_ode = pd.DataFrame({'time': [new_ode_time], 'height': [new_ode_ht]})
    new_rows_real = pd.DataFrame({'time': [new_real_time], 'height': [new_real_ht]})
    df_ode = pd.read_csv('cache_ode.csv')
    df_real = pd.read_csv('cache_real.csv')
    df_ode = pd.concat([df_ode, new_rows_ode], ignore_index=True)
    df_real = pd.concat([df_real, new_rows_real], ignore_index=True)
    df_ode.to_csv('cache_ode.csv', index=False)
    df_real.to_csv('cache_real.csv', index=False)
    
    c_ode = alt.Chart(df_ode).mark_line(color='blue').encode(
        x='time',
        y='height'
    ).properties(title='ODE Data')
    
    c_real = alt.Chart(df_real).mark_line(color='red').encode(
        x='time',
        y='height'
    ).properties(title='Real Data')
    
    c = alt.layer(c_ode, c_real).resolve_scale(y='independent')
    chart_container.altair_chart(c, use_container_width=True)


# Auto-update plot every 2 seconds
h_data = []


st.sidebar.header("Control Panel")

# Controller parameters
Kp = st.sidebar.number_input("Kp", min_value=0.0, max_value=10.0, value=1.0)
Ti = st.sidebar.number_input("Ti", min_value=0.0, max_value=10.0, value=1.0)
Td = st.sidebar.number_input("Td", min_value=0.0, max_value=10.0, value=1.0)
Ki = 1/Ti
Kd = 1/Td

# Root URL

# Vin
vin = st.sidebar.number_input("Vin / setpoint", min_value=0.0, max_value=10.0, value=1.0)

# Openloop or closedloop
is_openloop = st.sidebar.checkbox("Openloop", value=False)

# Start/Stop buttons
if st.sidebar.button("Start"):
    if system == "ODE":
        requests.get(f"{root_url}/ODE/")
    else:
        requests.get(f"{root_url}/real/?setpoint={setpoint}")

if st.sidebar.button("Stop"):
    if system == "ODE":
        requests.post(f"{root_url}/ODE/stop")
    else:
        requests.post(f"{root_url}/real/stop")

# Update parameters
if st.sidebar.button("Update Parameters"):
    if system == "ODE":
        requests.post(f"{root_url}/ODE/update_setpoint", json={"setpoint": setpoint})
        requests.post(f"{root_url}/ODE/update_ctrlr", json={"Kp": Kp, "Ki": Ki, "Kd": Kd})
        requests.post(f"{root_url}/ODE/update_vin", json={"vin": vin})
        requests.post(f"{root_url}/ODE/is_openloop", json={"is_openloop": is_openloop})
    else:
        requests.post(f"{root_url}/real/update_setpoint", json={"setpoint": setpoint})
        requests.post(f"{root_url}/real/update_ctrlr", json={"Kp": Kp, "Ki": Ki, "Kd": Kd})
        requests.post(f"{root_url}/real/update_vin", json={"vin": vin})
        requests.post(f"{root_url}/real/is_openloop", json={"is_openloop": is_openloop})

if st.sidebar.checkbox("Auto-update plot", value=False):
    while True:
        fetch_and_plot_data()
        time.sleep(2)
else:
    if st.sidebar.button("Fetch Data"):
        fetch_and_plot_data()
