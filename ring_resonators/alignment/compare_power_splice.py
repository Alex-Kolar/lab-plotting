import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DATA_FILE_OLD = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab members/ZhongLab_Alex"
                 "/Ring Resonators/Alignment Tests/Feedthrough Power/power_data_20240604_095547.csv")
DATA_FILE_FIX = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab members/ZhongLab_Alex/"
                 "Ring Resonators/Alignment Tests/Feedthrough Power/power_data_20240608_131556.csv")

# plotting params
color_fix = 'tab:blue'
color_old = 'tab:orange'


# data read and prep
df_fix = pd.read_csv(DATA_FILE_FIX)
df_old = pd.read_csv(DATA_FILE_OLD)

time_series = df_fix["Epoch Time (s)"]
time_series = time_series - time_series[0]
time_series /= 3600  # convert to hours

time_series_control = df_old["Epoch Time (s)"]
time_series_control = time_series_control - time_series_control[0]
time_series_control /= 3600  # convert to hours

power_series = df_fix["Power (uW)"]
power_series_control = df_old["Power (uW)"]


# plotting of power
fig, ax = plt.subplots()

ax.plot(time_series_control, power_series_control,
        color=color_old, label='Before Feedthrough Adjustment')
ax.plot(time_series, power_series,
        color=color_fix, label='After Feedthrough Adjustment')

ax.set_title("Power versus Time")
ax.set_xlabel("Time (hours)")
ax.set_ylabel(r"Power ($\mu$W)")
ax.grid(True)
ax.legend(shadow=True, loc='lower right')

# ax.set_ylim((0, 0.110))
# ax_r.set_ylim((0, 4500))
ax.set_xlim((0, 24))

fig.tight_layout()
fig.show()
