import pandas as pd
import matplotlib.pyplot as plt


DATA_FILE = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab members/ZhongLab_Alex"
             "/Ring Resonators/Alignment Tests/position_data_20240522_135248.csv")
SHOW_THRESH = True
MAX_THRESH = 0.100  # units: uW
MIN_THRESH = 0.010  # units: uW


df = pd.read_csv(DATA_FILE)
time_series = df["Epoch Time (s)"]
time_series = time_series - time_series[0]
time_series /= 3600  # convert to hours


# plotting of power
fig, ax = plt.subplots()

ax.plot(time_series, df["Power (uW)"])
if SHOW_THRESH:
    ax.axhline(MAX_THRESH, color='k', ls='--')
    ax.axhline(MIN_THRESH, color='k', ls='--')

    # add labels
    ax_r = ax.twinx()
    ax_r.set_ylim(ax.get_ylim())
    ax_r.set_yticks([MIN_THRESH, MAX_THRESH],
                    ["Min Power", "Max Power"])

ax.set_title("Power versus Time")
ax.set_xlabel("Time (hours)")
ax.set_ylabel(r"Power ($\mu$W)")
ax.grid(True)

ax.set_ylim((0, 0.110))
ax_r.set_ylim((0, 0.110))

fig.tight_layout()
fig.show()


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

ax[2].set_ylim((1400, 1500))

plt.tight_layout()
plt.show()
