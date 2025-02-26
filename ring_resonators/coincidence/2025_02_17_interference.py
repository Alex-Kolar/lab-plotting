import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# data params
DATA_1 = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
          "/New_mounted_device/300K_no_erbium/coincidence/02172025/correlation_5min_00V.txt")
DATA_2 = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
          "/New_mounted_device/300K_no_erbium/coincidence/02172025/correlation_5min_10V.txt")


# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
xlim = (-30, 30)
color_1 = 'cornflowerblue'
color_2 = 'coral'
alpha = 0.5

# extract data
df_1 = pd.read_csv(DATA_1, sep='\t')
df_2 = pd.read_csv(DATA_2, sep='\t')

coincidence_1 = df_1["Counts"]
time_1 = df_1["Time(ps)"]  # unit: ps
time_1 *= 1e-3  # unit: ns
time_diff_1 = time_1[1] - time_1[0]  # spacing of histogram

coincidence_2 = df_2["Counts"]
time_2 = df_2["Time(ps)"]  # unit: ps
time_2 *= 1e-3  # unit: ns
time_diff_2 = time_2[1] - time_2[0]  # spacing of histogram


# plotting
plt.bar(time_1, coincidence_1, width=time_diff_1,
        color=color_1, alpha=alpha,
        label='0 V')
plt.bar(time_2, coincidence_2, width=time_diff_2,
        color=color_2, alpha=alpha,
        label='10 V')

plt.title('Coincidence Count Interference')
plt.xlabel('Timing Offset (ns)')
plt.ylabel('Counts')
plt.legend(shadow=True)
plt.xlim(xlim)

plt.tight_layout()
plt.show()
