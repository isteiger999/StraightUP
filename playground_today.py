import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df_raw = pd.read_csv(r"data/beep_schedules_Mohid3/airpods_motion_1760174615.csv")
#df_delta = pd.read_csv(r"data/beep_schedules_Dario0/airpods_motion_d1760171174.csv")
#df_raw = pd.read_csv(r"data/beep_schedules_Claire0/airpods_motion_1760629578.csv")
df_delta = pd.read_csv(r"data/beep_schedules_Mohid3/airpods_motion_1760174615.csv")

length = 5000
chanel = "acc_y"
df_raw = df_raw[chanel]
df_delta = df_delta[chanel]
df_raw = df_raw.iloc[:length]
df_delta = df_delta.iloc[:length]

plt.plot(np.arange(length), df_raw, label = "Dario delta")
plt.plot(np.arange(length), df_delta, label = "Claire delta")
plt.plot(np.arange(length), np.zeros(length))
plt.xlabel("Frames")
plt.title(chanel)
plt.ylabel(chanel)
plt.legend()
plt.show()