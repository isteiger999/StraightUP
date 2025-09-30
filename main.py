import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#data_test = pd.read_csv(r"Data_Airpods/airpods_motion_1.csv", na_values=['NA'])
data_test = pd.read_csv(r"Data_Airpods/airpods_motion_2.csv", na_values=['NA'])
offset_time = data_test.iloc[0, 0]
data_test['timestamp'] = data_test['timestamp'] - offset_time
print(offset_time)

acc_x = data_test['acc_x']
acc_y = data_test['acc_y']
acc_z = data_test['acc_z']
acc_tot = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

plt.plot(data_test['timestamp'], acc_tot, label='acc_tot')
plt.plot(data_test['timestamp'], acc_x, label='acc_x')
plt.plot(data_test['timestamp'], acc_y, label='acc_y')
plt.plot(data_test['timestamp'], acc_z, label='acc_z')
plt.xlabel("Time")
plt.ylabel("Acceleration in x")
plt.legend()
plt.show()