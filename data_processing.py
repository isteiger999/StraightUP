import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path   # to cycle through csv files in folders

def drop_timestamp_inplace(folders):
    for folder in map(Path, folders):
        for csv_path in folder.glob("*.csv"):
            df = pd.read_csv(csv_path, low_memory=False)
            if "timestamp" in df.columns:
                df = df.drop(columns=["timestamp"])
                df.to_csv(csv_path, index=False)
            else:
                pass

def plot_data():
    data_test = pd.read_csv(r"slouch_data/slouch0.csv", na_values=['NA'])

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

    plt.plot(data_test['timestamp'].iloc[-500:], data_test['grav_y'].iloc[-500:], label='grav_y')
    plt.xlabel("Time")
    plt.ylabel("Acceleration in x")
    plt.title('total recording')
    plt.legend()
    plt.show()

def obtain_windows():
    f_sample = 50      # [Hz].  (actually 50.07 Hz) 180sekunden --> 12.6 frames = 0.252s
    window_length = 8  # [sec]
    timestep_window = f_sample * window_length
    windows_tot = 22
    samples_slouch = 5 * 11
    samples_noslouch = 3 * 22
    samples_tot = samples_slouch + samples_noslouch
    chanels = 13

    X_tot = np.zeros((samples_tot, timestep_window, chanels)) # shape: 50'000, timesteps(8*25), channels(13)
    y_tot = np.zeros((samples_tot, 1))


    ## for SLOUCH data ##
    slouch_count = 0
    folder = Path("slouch_data")
    for rec in folder.glob("*.csv"):
        df = pd.read_csv(rec, na_values=['NA'])    
        for index in range(windows_tot):
            if index % 2 != 0 and index != 0:
                X_tot[slouch_count, :, :] = df.iloc[index*timestep_window:(index+1)*timestep_window, :].values
                y_tot[slouch_count] = 1
                slouch_count += 1

    ## for NO_SLOUCH data ##. (can use all 22 windows instead of only 11)
    folder = Path("no_slouch_data")
    no_slouch_count = 0
    for rec in folder.glob("*.csv"):
        df = pd.read_csv(rec, na_values=['NA'])      
        for index in range(windows_tot):
            X_tot[slouch_count + index, :, :] = df.iloc[index*timestep_window:(index+1)*timestep_window, :].values
            y_tot[slouch_count + index] = 0
            no_slouch_count += 1

    print(f"X_tot shape: {X_tot.shape}; y_tot shape: {y_tot.shape}")
    
    return X_tot, y_tot