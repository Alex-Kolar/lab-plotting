import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DATA_FILE_CONTROL = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab members/ZhongLab_Alex"
                     "/Ring Resonators/Alignment Tests/Feedthrough Power/power_data_20240604_095547.csv")
DATA_FILE = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab members/ZhongLab_Alex"
             "/Ring Resonators/Alignment Tests/position_data_20240522_135248.csv")

# plotting params
color = 'tab:blue'
color_control = 'tab:orange'


# data read and prep
df = pd.read_csv(DATA_FILE)
df_control = pd.read_csv(DATA_FILE_CONTROL)

time_series = df["Epoch Time (s)"]
time_series = time_series - time_series[0]
time_series /= 3600  # convert to hours

time_series_control = df_control["Epoch Time (s)"]
time_series_control = time_series_control - time_series_control[0]
time_series_control /= 3600  # convert to hours

power_series = df["Power (uW)"]
power_series_control = df_control["Power (uW)"]


# plotting of power
fig, ax = plt.subplots()
ax_r = ax.twinx()

ax.plot(time_series, power_series,
        color=color)
ax_r.plot(time_series_control, power_series_control,
          color=color_control)

ax.tick_params(axis='y', colors=color)
ax_r.tick_params(axis='y', colors=color_control)
ax.set_title("Power versus Time")
ax.set_xlabel("Time (hours)")
ax.set_ylabel(r"Power ($\mu$W)",
              color=color)
ax_r.set_ylabel(r"Power ($\mu$W)",
                color=color_control)
ax.grid(True)

ax.set_ylim((0, 0.110))
ax_r.set_ylim((0, 4500))

fig.tight_layout()
fig.show()
