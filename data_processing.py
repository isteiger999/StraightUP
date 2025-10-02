import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal


def plot_data():
    data_test = pd.read_csv(r"Data_Airpods/airpods_motion_erster_test.csv", na_values=['NA'])

    offset_time = data_test.iloc[0, 0]
    data_test['timestamp'] = data_test['timestamp'] - offset_time

    acc_x = data_test['acc_x']
    acc_y = data_test['acc_y']
    acc_z = data_test['acc_z']
    acc_tot = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

    grav_x = data_test['grav_x']
    grav_y = data_test['grav_y']
    grav_z = data_test['grav_z']
    grav_tot = np.sqrt(grav_x**2 + grav_y**2 + grav_z**2)

    #plt.plot(data_test['timestamp'], acc_tot, label='acc_tot')
    #plt.plot(data_test['timestamp'], acc_x, label='acc_x')
    #plt.plot(data_test['timestamp'], acc_y, label='acc_y')
    #plt.plot(data_test['timestamp'], acc_x, label='acc_x')
    #plt.plot(data_test['timestamp'], acc_y, label='acc_y')
    #plt.plot(data_test['timestamp'], acc_z, label='acc_z')
    #plt.plot(data_test['timestamp'], data_test['grav_x'], label='grav_x')
    plt.plot(data_test['timestamp'].iloc[-1000:], data_test['grav_y'].iloc[-1000:], label='grav_y')
    #plt.plot(data_test['timestamp'], data_test['grav_z'], label='grav_z')
    plt.xlabel("Time")
    plt.ylabel("Acceleration in x")
    plt.legend()
    plt.show()

def obtain_windows():
    f_sample = 50      # [Hz].  (actually 50.07 Hz) 180sekunden --> 12.6 frames = 0.252s
    window_length = 8  # [sec]
    timestep_window = f_sample * window_length
    windows_tot = 22
    samples = 11
    chanels = 14

    rec = pd.read_csv(r"Data_Airpods/airpods_motion_erster_test.csv", na_values=['NA'])
    offset_time = rec.iloc[0, 0]
    rec['timestamp'] = rec['timestamp'] - offset_time

    X_train = np.zeros((samples, timestep_window, chanels)) # shape: 50'000, timesteps(8*25), channels(13)
    y_train = np.ones((chanels, 1))

    i = 0
    for index in range(windows_tot):
        if index % 2 != 0 and index != 0:
            X_train[i, :, :] = rec.iloc[index*timestep_window:(index+1)*timestep_window, :].values
            i += 1
            
    plt.plot(np.arange(400), X_train[10, :, 12], label='grav_y')
    plt.xlabel("Time")
    plt.ylabel("Acceleration in x")
    plt.legend()
    plt.show()



    return X_train, y_train
    