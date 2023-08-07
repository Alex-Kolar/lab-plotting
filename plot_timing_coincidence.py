import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


FILENAME = "/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators" \
           "/Coincidence Count Measurement/08022023/Correlation-2_2023-08-03_14-41-25_(30sec_int).txt"

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
xlim = (-50, 50)
color = 'cornflowerblue'

# NOTE: first line of csv needs extra tab added
# otherwise the columns will not be read properly
df = pd.read_csv(FILENAME, sep='\t')

# convert time
time = df["Time(ps)"]  # unit: ps
time *= 1e-3  # unit: ns
time_diff = time[1] - time[0]


# plotting
plt.bar(time, df["Counts"], width=time_diff, color=color)
plt.xlim(xlim)
plt.xlabel("Timing Offset (ns)")
plt.ylabel("Coincidence Counts")

plt.tight_layout()
plt.show()
