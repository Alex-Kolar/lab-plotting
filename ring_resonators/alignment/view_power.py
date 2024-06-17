import pandas as pd
import matplotlib.pyplot as plt


DATA_FILE = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab members/ZhongLab_Alex"
             "/Ring Resonators/Alignment Tests/Feedthrough Power/power_data_20240604_095547.csv")


# data read and prep
df = pd.read_csv(DATA_FILE)

time_series = df["Epoch Time (s)"]
time_series = time_series - time_series[0]
time_series /= 3600  # convert to hours

power_series = df["Power (uW)"]
power_series /= 1e3  # convert to mW


# plotting of power
fig, ax = plt.subplots()

ax.plot(time_series, power_series)

ax.set_title("Power versus Time")
ax.set_xlabel("Time (hours)")
ax.set_ylabel(r"Power (mW)")
ax.grid(True)

fig.tight_layout()
fig.show()
