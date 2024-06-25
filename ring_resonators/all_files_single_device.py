"""Read all files in a given directory and fit them.

Saves output graph as well.
"""

import glob
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import BreitWignerModel, LinearModel


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/Mounted_device/04292024")
OUTPUT_DIR = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
              "/mounted/04292024")
SCAN_RANGE = 13.5  # unit: GHz

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'

PlOT_ALL_RES = True


# get scan data
filenames = glob.glob('SDS*.csv', recursive=True, root_dir=DATA_DIR)
paths = [os.path.join(DATA_DIR, file) for file in filenames]
dfs = [pd.read_csv(file, header=10, skiprows=[11]) for file in paths]
# get frequency data
filename = os.path.join(DATA_DIR, "frequencies.csv")
df_freq = pd.read_csv(filename)

# get scan number and sort
numbers = []
for file in filenames:
    num_str = file[:-4]  # remove '.csv'
    num_str = num_str[3:]  # remove 'SDS'
    numbers.append(int(num_str))
dfs = [df for _, df in sorted(zip(numbers, dfs))]
numbers.sort()


# main data processing loop
model = BreitWignerModel() + LinearModel()
all_q = []

# do data fitting
all_q = []
for num, df in zip(numbers, dfs):
    center_freq = df_freq["Frequency (THz)"][num - 1]
    ramp = df['CH1'].astype(float)
    transmission = df['CH2'].astype(float)

    id_min = np.argmin(ramp)
    id_max = np.argmax(ramp)
    transmission = transmission[id_min:id_max]
    freq = np.linspace(0, SCAN_RANGE * 1e3, id_max - id_min)  # unit: MHz

    out = model.fit(transmission, x=freq,
                    center=7000, amplitude=0.02, sigma=200,
                    slope=0, intercept=max(transmission))

    freq_light = center_freq * 1e6  # units: MHz
    q = freq_light / out.params['sigma'].value
    print(f"\t{q}")
    all_q.append(q)

    if PlOT_ALL_RES:
        plt.plot(freq, transmission, color=color)
        plt.plot(freq, out.best_fit, 'k--')

        plt.text(out.params["center"].value + 100,
                 min(transmission),
                 f"Q: {q:.3}")

        plt.xlim((out.params["center"].value - 1000, out.params["center"].value + 1000))

        plt.title(f"{center_freq} THz scan")
        plt.xlabel("Detuning (MHz)")
        plt.ylabel("Transmission (A.U.)")
        plt.grid(True)

        plt.tight_layout()

        save_name = f"Scan_{num}_{center_freq}_THz"
        save_name = save_name.replace(".", "_")
        save_name += ".png"
        plt.savefig(os.path.join(OUTPUT_DIR, save_name))
        plt.clf()

print(f"Max Q: {max(all_q)} (Scan {np.argmax(all_q)+1})")

# plt.bar(range(len(device_set)), q_max,
#         color=color, edgecolor='k', zorder=2)
#
# plt.xticks(range(len(device_set)), device_set)
# plt.xlabel("Device Number")
# plt.ylabel("Highest Q Factor")
# plt.grid(axis='y')
#
# plt.tight_layout()
# plt.show()
