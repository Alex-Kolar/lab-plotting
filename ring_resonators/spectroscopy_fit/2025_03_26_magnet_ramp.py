import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/10mK/03262025")
DATA_FILES = ["PRERAMP.csv",
              "RAMP1.csv",
              "RAMP2.csv",
              "POSTRAMP.csv"]
FIELDS = [214, 166, 114, 100]  # unit: mT
FREQ_RANGE = (194807.084, 194815.654)  # unit: GHz
AOM_OFFSET = 0.600  # unit: GHz

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
cmap = 'Greens'
ref_freq = 194807  # unit: GHz


# read data files
dfs = []
for file in DATA_FILES:
    path = os.path.join(DATA_DIR, file)
    dfs.append(pd.read_csv(path, header=10, skiprows=[11]))

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


# # plot all (for reference)
# for transmission in transmissions:
#     plt.plot(freq, transmission)
#
# plt.tight_layout()
# plt.show()


# plot all (waterfall, better)

