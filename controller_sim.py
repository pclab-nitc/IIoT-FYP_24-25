import math as m
import time
import threading
import queue
import csv
from datetime import datetime as dt
from scipy.integrate import odeint
from simple_pid import PID
import serial


class DifferentialEqnThread(threading.Thread):
    def __init__(self, set_point_height : queue.Queue, q_deq=None):
        super().__init__()
        self.stop_sim = False
        self.get_set_point_height = set_point_height
        self.q_deq = q_deq
        self.hist = []

    def run(self):
        def fp_model(h, t, v):
            """
            Tank flow process model.
            
            Parameters:
            h (float): Current height of the tank liquid (m).
            t (float): Current time (s).
            v_func (function): Interpolated voltage function.
            
            Returns:
            float: Rate of change of liquid height (dh/dt).
            """
            # Constants

            PI = m.pi
            d = 0.008  # Orifice diameter (m)
            r = 0.185  # Tank radius (m)
            h0 = 0.025  # Reference height (m)
            cf = 0.375 # Discharge coefficient

            # Interpolated voltage at time t
            

            # Flowrate calculation
            f = (4.042 * v - 2.866) / 1000000

            # Prevent negative or zero height difference
            h_effective = max(h - h0, 0)

            # Avoid division by zero in tank geometry calculation
            area = PI * (2 * r * h - h**2)
            if area <= 0:
                return 0  # No change if area is invalid

            # Differential equation for height change
            dhdt = (f - (cf * (PI * d**2)/4 * m.sqrt(2 * 9.81 * h_effective))) / area
            return dhdt

        h_current = 0.04
        pid = PID(20.0, 1, 0)
        pid.output_limits = (0, 12)
        start_time = time.time()
        prev_time =start_time

        while not self.stop_sim:
            time.sleep(1)
            del_t = prev_time-time.time()
            pid.setpoint = self.get_set_point_height.get()
            v = pid(h_current)
            t = [0.0, del_t]
            h = odeint(fp_model, h_current, t, args=(v,))
            h_current = h[-1][0]
            elapsed_time = time.time() - start_time
            self.q_deq.put((h_current, elapsed_time))
            print(h_current,elapsed_time,self.get_set_point_height.get())
            self.hist.append((elapsed_time, h_current))
            

    def stop(self):
        self.stop_sim = True
        fields = ['time', 'height']
        rows = self.hist
        with open(f"dataset-folder/deq_data{dt.isoformat(dt.now())[:-10]}.csv", 'w') as f:
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(rows)


class RealSystemThread(threading.Thread):
    def __init__(self, set_point_height:queue.Queue, q_real=None, mode="1"):
        super().__init__()
        self.stop_sim = False
        self.get_set_point_height = set_point_height
        self.q_real = q_real
        self.mode = mode
        self.hist = []

    def run(self):
        
        SERIAL_PORT = '/dev/ttyACM0'
        BAUD_RATE = 9600
        self.arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        pid = PID(20.0, 1, 0)
        pid.output_limits = (0, 12)
        start_time = time.time()
        h = 0.0401
        hprev = h

        while not self.stop_sim:
            time.sleep(1)
            pid.setpoint = self.get_set_point_height.get()
            if self.arduino.in_waiting > 0:
                data = self.arduino.readline().decode('utf-8').strip()
                try:
                    height = data.split()[-1]
                    h = 0.178 - float(height) / 100
                    if abs(h - hprev) > 0.04:
                        h = hprev

                    elapsed_time = time.time() - start_time
                    self.q_real.put((h, elapsed_time))
                    self.hist.append((elapsed_time, h))

                    op = pid(h)
                    if self.arduino.is_open:
                        pwm = int(1023 * op / 12)
                        self.arduino.write(f"{pwm}\n".encode())
                    hprev = h

                except Exception as e:
                    print(f"Invalid data from Arduino: {e}")

    def stop(self):
        self.stop_sim = True
        fields = ['time', 'height']
        rows = self.hist
        with open(f"dataset-folder/real_data{dt.isoformat(dt.now())[:-10]}.csv", 'w') as f:
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(rows)


class PINNModelThread(threading.Thread):
    def __init__(self, set_point_height, q_pinn=None):
        super().__init__()
        self.stop_sim = False
        self.get_set_point_height = set_point_height
        self.q_pinn = q_pinn

    def run(self):
        start_time = time.time()
        h_current = 0.025

        while not self.stop_sim:
            time.sleep(1)
            elapsed_time = time.time() - start_time
            self.q_pinn.put((h_current, elapsed_time))
            # print(h_current)

    def stop(self):
        self.stop_sim = True


def ProcessStart(set_point_height=0.025):
    q_deq = queue.Queue()
    q_real = queue.Queue()
    q_pinn = queue.Queue()

    deq_thread = DifferentialEqnThread(set_point_height=set_point_height, q_deq=q_deq)
    real_thread = RealSystemThread(set_point_height=set_point_height, q_real=q_real)
    pinn_thread = PINNModelThread(set_point_height=set_point_height, q_pinn=q_pinn)

    deq_thread.start()
    # real_thread.start()
    pinn_thread.start()

    return q_deq, q_real, q_pinn
