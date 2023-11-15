import glob
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit.models import BreitWignerModel, LinearModel


DATA_DIR = "/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators" \
           "/Planarized_device/long_scan_11132023"

# data taken for frequency
F_RANGE = 30.623  # units: GHz
F_CENTER = [195058, 195083, 195108, 195133, 195158, 195183, 195208, 195233, 195258, 195033, 195008, 194983, 194958, 194933, 194908, 194883, 194858]

# fitting parameters


# plotting parameters
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
figsize = (18, 6)


# locate all files
csv_files = glob.glob('*.csv', root_dir=DATA_DIR)
csv_paths = [os.path.join(DATA_DIR, file) for file in csv_files]

# sort all files and get scan number
csv_paths = sorted(csv_paths)
dfs = [pd.read_csv(path, header=5) for path in csv_paths]

# truncate data based on frequency scan
all_freq = []
all_trans = []
detune_center = F_CENTER[0]
for df, center in zip(dfs, F_CENTER):
    scan = df['Volt.1']
    id_min = scan.idxmin()
    id_max = scan.idxmax()
    freq = np.linspace((center-F_RANGE/2) - detune_center,
                       (center+F_RANGE/2) - detune_center,
                       id_max - id_min)
    trans = df['Volt'][id_min:id_max]

    all_freq.append(freq)
    all_trans.append(trans)


# plotting
fig, ax = plt.subplots(figsize=figsize)
for trans, freq in zip(all_trans, all_freq):
    ax.plot(freq, trans, color=color)

ax.grid('on')
ax.set_xlabel("Detuning (GHz)")
ax.set_ylabel("Transmission")

fig.tight_layout()
fig.show()


for i, (df, center) in enumerate(zip(dfs, F_CENTER)):
    scan = df['Volt.1']
    id_min = scan.idxmin()
    id_max = scan.idxmax()
    freq = np.linspace(-F_RANGE/2, F_RANGE/2,
                       id_max - id_min)
    trans = df['Volt'][id_min:id_max]
    plt.plot(freq, trans, color=color)

    plt.title(f"{i+1} scan ({center} GHz)")
    plt.tight_layout()
    plt.show()
