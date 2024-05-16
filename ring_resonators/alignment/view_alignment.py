import pandas as pd
import matplotlib.pyplot as plt


DATA_FILE = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab members/ZhongLab_Alex"
             "/Ring Resonators/Alignment Tests/position_data_20240404_095511.csv")


df = pd.read_csv(DATA_FILE)
time_series = df["Epoch Time (s)"]
time_series = time_series - time_series[0]
time_series /= 3600  # convert to hours


# plotting of power
plt.plot(df["Power (uW)"])
plt.plot(time_series, df["Power (uW)"])

plt.title("Power versus Time")
plt.xlabel("Time (hours)")
plt.ylabel(r"Power ($\mu$W)")
plt.grid(True)

plt.tight_layout()
plt.show()


# plotting of Relative position
fig, ax = plt.subplots(3, 1)
ax[0].plot(time_series, df["Current position axis 0"], color='tab:blue')
ax[1].plot(time_series, df["Current position axis 1"], color='tab:orange')
ax[2].plot(time_series, df["Current position axis 2"], color='tab:green')

ax[0].set_title("Relative Position (steps) versus Time")
ax[2].set_xlabel("Time (hours)")
ax[0].set_ylabel("Axis 0")
ax[1].set_ylabel("Axis 1")
ax[2].set_ylabel("Axis 2")

plt.tight_layout()
plt.show()


# plotting of Real position
# multiply by 1e6 to convert to micrometers
fig, ax = plt.subplots(3, 1)
ax[0].plot(time_series, df["Real position axis 0"]*1e6, color='tab:blue')
ax[1].plot(time_series, df["Real position axis 1"]*1e6, color='tab:orange')
ax[2].plot(time_series, df["Real position axis 2"]*1e6, color='tab:green')

ax[0].set_title(r"Real Position ($\mu$m) versus Time")
ax[2].set_xlabel("Time (hours)")
ax[0].set_ylabel("Axis 0")
ax[1].set_ylabel("Axis 1")
ax[2].set_ylabel("Axis 2")

ax[1].set_ylim((0.0035, 0.0037))

plt.tight_layout()
plt.show()
