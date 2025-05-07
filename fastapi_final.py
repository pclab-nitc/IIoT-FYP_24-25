from typing import Union

from fastapi import FastAPI
import sqlite3
import pandas as pd
from fastapi.responses import JSONResponse

from collections import deque
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PyQt5.QtCore import QThread
from qtask import DifferentialEqnThread, RealSystemThread, TransferFunctionModelThread, DataDrivenModelThread
from collections import deque
import sqlite3
import pandas as pd

app = FastAPI()

# Shared data structures
ode_height = deque(maxlen=50)
real_height = deque(maxlen=50)
tf_height = deque(maxlen=50)
dd_height = deque(maxlen=50)

# Initialize simulation threads
ode_thread = DifferentialEqnThread(set_point_height=0.025, kp=30.0, ki=1.0, kd=0.0, open_loop=True)
real_thread = RealSystemThread(set_point_height=0.025, kp=30.0, ki=1.0, kd=0.0, open_loop=True)
tf_thread = TransferFunctionModelThread(set_point_height=0.025, kp=30.0, ki=1.0, kd=0.0, open_loop=True)
dd_thread = DataDrivenModelThread(set_point_height=0.025, kp=30.0, ki=1.0, kd=0.0, open_loop=True)

DB_PATH = "history.db"

"""-----------------ODE----------------"""

@app.get("/ODE/")
def start_ODE():
    if not ode_thread.isRunning():
        ode_thread.start()
        return {"status": "ODE simulation started"}
    return {"status": "ODE simulation already running"}

@app.post("/ODE/update_setpoint")
def update_setpoint_ODE(setpoint: float):
    ode_thread.set_point_height = setpoint
    return {"message": f"Setpoint updated to {setpoint}"}

@app.post("/ODE/update_ctrlr")
def update_ctrlr_ODE(kp: float, ki: float, kd: float):
    ode_thread.pid.Kp = kp
    ode_thread.pid.Ki = ki
    ode_thread.pid.Kd = kd
    return {"message": f"Controller updated to Kp={kp}, Ki={ki}, Kd={kd}"}

@app.post("/ODE/update_vin")
def update_vin_ODE(vin: float):
    ode_thread.set_point_height = vin  # Assuming vin is equivalent to set_point_height in open-loop mode
    return {"message": f"Vin updated to {vin}"}

@app.post("/ODE/is_openloop")
def is_openloop_ODE(is_openloop: bool):
    ode_thread.open_loop = is_openloop
    return {"message": f"Openloop status updated to {is_openloop}"}

@app.get("/ODE/status")
def get_status_ODE():
    return {
        "Kp": ode_thread.pid.Kp,
        "Ki": ode_thread.pid.Ki,
        "Kd": ode_thread.pid.Kd,
        "Vin": ode_thread.set_point_height,
        "Setpoint": ode_thread.set_point_height,
        "ctrlr_mode": "openloop" if ode_thread.open_loop else "closedloop",
        "running": ode_thread.isRunning(),
    }

@app.post("/ODE/stop")
def stop_ODE():
    ode_thread.stop()
    return {"message": "ODE simulation stopped"}

"""------------Real System-------------"""

@app.get("/real/")
def start_real():
    if not real_thread.isRunning():
        real_thread.start()
        return {"status": "Real system simulation started"}
    return {"status": "Real system simulation already running"}

@app.post("/real/update_setpoint")
def update_setpoint_real(setpoint: float):
    real_thread.set_point_height = setpoint
    return {"message": f"Setpoint updated to {setpoint}"}

@app.post("/real/update_ctrlr")
def update_ctrlr_real(kp: float, ki: float, kd: float):
    real_thread.pid.Kp = kp
    real_thread.pid.Ki = ki
    real_thread.pid.Kd = kd
    return {"message": f"Controller updated to Kp={kp}, Ki={ki}, Kd={kd}"}

@app.post("/real/update_vin")
def update_vin_real(vin: float):
    real_thread.set_point_height = vin  # Assuming vin is equivalent to set_point_height in open-loop mode
    return {"message": f"Vin updated to {vin}"}

@app.post("/real/is_openloop")
def is_openloop_real(is_openloop: bool):
    real_thread.open_loop = is_openloop
    return {"message": f"Openloop status updated to {is_openloop}"}

@app.get("/real/status")
def get_status_real():
    return {
        "Kp": real_thread.pid.Kp,
        "Ki": real_thread.pid.Ki,
        "Kd": real_thread.pid.Kd,
        "Vin": real_thread.set_point_height,
        "Setpoint": real_thread.set_point_height,
        "ctrlr_mode": "openloop" if real_thread.open_loop else "closedloop",
        "running": real_thread.isRunning(),
    }

@app.post("/real/stop")
def stop_real():
    real_thread.stop()
    return {"message": "Real system simulation stopped"}

"""------------Transfer Function Model-------------"""

@app.get("/tfm/")
def start_tfm():
    if not tf_thread.isRunning():
        tf_thread.start()
        return {"status": "Transfer function model simulation started"}
    return {"status": "Transfer function model simulation already running"}

@app.post("/tfm/update_setpoint")
def update_setpoint_tfm(setpoint: float):
    tf_thread.set_point_height = setpoint
    return {"message": f"Setpoint updated to {setpoint}"}

@app.post("/tfm/update_ctrlr")
def update_ctrlr_tfm(kp: float, ki: float, kd: float):
    tf_thread.pid.Kp = kp
    tf_thread.pid.Ki = ki
    tf_thread.pid.Kd = kd
    return {"message": f"Controller updated to Kp={kp}, Ki={ki}, Kd={kd}"}

@app.post("/tfm/update_vin")
def update_vin_tfm(vin: float):
    tf_thread.set_point_height = vin  # Assuming vin is equivalent to set_point_height in open-loop mode
    return {"message": f"Vin updated to {vin}"}

@app.post("/tfm/is_openloop")
def is_openloop_tfm(is_openloop: bool):
    tf_thread.open_loop = is_openloop
    return {"message": f"Openloop status updated to {is_openloop}"}

@app.get("/tfm/status")
def get_status_tfm():
    return {
        "Kp": tf_thread.pid.Kp,
        "Ki": tf_thread.pid.Ki,
        "Kd": tf_thread.pid.Kd,
        "Vin": tf_thread.set_point_height,
        "Setpoint": tf_thread.set_point_height,
        "ctrlr_mode": "openloop" if tf_thread.open_loop else "closedloop",
        "running": tf_thread.isRunning(),
    }

@app.post("/tfm/stop")
def stop_tfm():
    tf_thread.stop()
    return {"message": "Transfer function model simulation stopped"}

"""------------Data Driven Model-------------"""

@app.get("/ddm/")
def start_ddm():
    if not dd_thread.isRunning():
        dd_thread.start()
        return {"status": "Data-driven model simulation started"}
    return {"status": "Data-driven model simulation already running"}

@app.post("/ddm/update_setpoint")
def update_setpoint_ddm(setpoint: float):
    dd_thread.set_point_height = setpoint
    return {"message": f"Setpoint updated to {setpoint}"}

@app.post("/ddm/update_ctrlr")
def update_ctrlr_ddm(kp: float, ki: float, kd: float):
    dd_thread.pid.Kp = kp
    dd_thread.pid.Ki = ki
    dd_thread.pid.Kd = kd
    return {"message": f"Controller updated to Kp={kp}, Ki={ki}, Kd={kd}"}

@app.post("/ddm/update_vin")
def update_vin_ddm(vin: float):
    dd_thread.set_point_height = vin  # Assuming vin is equivalent to set_point_height in open-loop mode
    return {"message": f"Vin updated to {vin}"}

@app.post("/ddm/is_openloop")
def is_openloop_ddm(is_openloop: bool):
    dd_thread.open_loop = is_openloop
    return {"message": f"Openloop status updated to {is_openloop}"}

@app.get("/ddm/status")
def get_status_ddm():
    return {
        "Kp": dd_thread.pid.Kp,
        "Ki": dd_thread.pid.Ki,
        "Kd": dd_thread.pid.Kd,
        "Vin": dd_thread.set_point_height,
        "Setpoint": dd_thread.set_point_height,
        "ctrlr_mode": "openloop" if dd_thread.open_loop else "closedloop",
        "running": dd_thread.isRunning(),
    }

@app.post("/ddm/stop")
def stop_ddm():
    dd_thread.stop()
    return {"message": "Data-driven model simulation stopped"}

@app.get("/history/{model_name}")
def get_history(model_name: str):
    """
    Fetch historical data (time, height, voltage) from the database for a given model.

    Args:
        model_name (str): The name of the model (e.g., "ODEsim", "RealSystem", "TransferFunctionModel").

    Returns:
        JSONResponse: A JSON object containing the historical data.
    """
    # Validate model name
    valid_models = ["ODEsim", "RealSystem", "TransferFunctionModel","DataDriven"]
    if model_name not in valid_models:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid model name. Choose from {valid_models}."},
        )

    # Connect to the database
    try:
        conn = sqlite3.connect(DB_PATH)
        query = f"SELECT time, height, voltage FROM {model_name}"
        df = pd.read_sql_query(query, conn)
        conn.close()

        # Convert the DataFrame to a list of dictionaries
        data = df.to_dict(orient="records")
        return {"model": model_name, "data": data}

    except sqlite3.Error as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Database error: {e}"},
        )