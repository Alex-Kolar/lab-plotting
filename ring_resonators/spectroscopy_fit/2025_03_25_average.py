import glob
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# collected data
DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/10mK/s2s_03252025")
FREQ_RANGE = (194807.649, 194816.359)
AOM_OFFSET = 0.600  # unit: GHz

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color_1 = 'cornflowerblue'
color_2 = 'coral'
color_3 = 'mediumpurple'
ref_freq = 194807  # unit: GHz


files = glob.glob(os.path.join(DATA_DIR, "PREINIT*.csv"))
dfs = [pd.read_csv(file, header=10, skiprows=[11]) for file in files]

# get ramp data from first file
df_start = dfs[0]
ramp = df_start['CH1'].astype(float).to_numpy()
id_min = np.argmin(ramp)
id_max = np.argmax(ramp)

# convert time to frequency
freq = np.linspace(FREQ_RANGE[0]-ref_freq, FREQ_RANGE[1]-ref_freq, id_max - id_min)  # unit: GHz
freq += AOM_OFFSET

# get transmission data for each scan
transmissions = []
for df in dfs:
    transmission = df['CH2']
    transmisison = transmission[id_min:id_max]
    transmissions.append(transmisison)


# plot all
for transmission in transmissions:
    plt.plot(freq, transmission)

plt.tight_layout()
plt.show()


# plot average
transmissions = np.array(transmissions)
avg = np.sum(transmissions, axis=0) / len(transmissions)
plt.plot(freq, avg)

plt.tight_layout()
plt.show()


# # plot first
# data = os.path.join(DATA_DIR, "PREINIT3.csv")
# df = pd.read_csv(data, header=10, skiprows=[11])
#
#
# fig, ax = plt.subplots()
# ax2 = ax.twinx()
#
# ax.plot(df['CH1'])
# ax2.plot(df['CH2'])
#
# fig.tight_layout()
# fig.show()
