import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal


#data_test = pd.read_csv(r"Data_Airpods/airpods_motion_1.csv", na_values=['NA'])
#data_test = pd.read_csv(r"Data_Airpods/airpods_motion_5.csv", na_values=['NA'])
#data_test = pd.read_csv(r"Data_Airpods/pure_x_seitwaerts.csv", na_values=['NA'])
data_test = pd.read_csv(r"Data_Airpods/airpods_motion_gravity.csv", na_values=['NA'])

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
plt.plot(data_test['timestamp'], data_test['grav_x'], label='grav_x')
plt.plot(data_test['timestamp'], data_test['grav_y'], label='grav_y')
plt.plot(data_test['timestamp'], data_test['grav_z'], label='grav_z')
plt.xlabel("Time")
plt.ylabel("Acceleration in x")
plt.legend()
plt.show()

## get windows
f_sample = 25      # [Hz]
window_length = 8  # [sec]
windows = 10
timesteps = f_sample * window_length * windows

X_train = # shape: 50'000, timesteps(8*25), channels(13)
index = 0
for timestep in range(timesteps):
    X_train[index, :, :] = rec[timestep:timestep + timesteps]
    timestep += timesteps
    index += 1