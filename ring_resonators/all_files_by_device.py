import glob
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import BreitWignerModel, LinearModel


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/Planarized_device/warmup_12122023/PRESSURE")
OUTPUT_DIR = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
              "/planarized/warmup_12122023/pressure")
SCAN_RANGE = 2  # unit: GHz

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'mediumpurple'

PlOT_ALL_RES = True


# find files and sort by device
filenames = glob.glob('*.csv', root_dir=DATA_DIR)
device_numbers = [int(fn[1:3]) for fn in filenames]
device_set = set(device_numbers)
filenames_dict = {device: [] for device in device_set}
for device, fn in zip(device_numbers, filenames):
    filenames_dict[device].append(fn)

# read data
df_dict = {}
for device in device_set:
    paths = [os.path.join(DATA_DIR, file) for file in filenames_dict[device]]
    df_dict[device] = [pd.read_csv(file, header=10, skiprows=[11]) for file in paths]


# main data processing loop
model = BreitWignerModel() + LinearModel()
q_max = []
for device in device_set:

    print(f"Device {device}:")

    dfs = df_dict[device]
    filenames = filenames_dict[device]

    # find wavelengths
    wavelengths = []
    for file in filenames:
        wl_str = file[:-4]  # remove '.csv'
        wl_str = wl_str[4:]  # remove 'DXX_'
        wl_str = wl_str.replace('_', '.')
        wavelengths.append(float(wl_str))

    # do data fitting
    all_q = []
    for wl, df in zip(wavelengths, dfs):
        ramp = df['CH1'].astype(float)
        transmission = df['CH2'].astype(float)

        id_min = np.argmin(ramp)
        id_max = np.argmax(ramp)
        transmission = transmission[id_min:id_max]
        freq = np.linspace(0, SCAN_RANGE * 1e3, id_max - id_min)  # unit: MHz

        out = model.fit(transmission, x=freq,
                        center=1000, amplitude=0.02, sigma=200,
                        slope=0, intercept=max(transmission))

        freq_light = (3e8 / (wl * 1e-9)) * 1e-6  # unit: MHz
        q = freq_light / out.params['sigma'].value
        print(f"\t{q}")
        all_q.append(q)

        if PlOT_ALL_RES:
            plt.plot(freq, transmission, color=color)
            plt.plot(freq, out.best_fit, 'k--')

            plt.text(out.params["center"].value + 100,
                     min(transmission),
                     f"Q: {q:.3}")

            plt.title(f"Device {device} {wl} nm scan")
            plt.xlabel("Detuning (MHz)")
            plt.ylabel("Transmission (A.U.)")
            plt.grid(True)

            plt.tight_layout()

            save_name = f"D{device}_{wl}"
            save_name = save_name.replace(".", "_")
            save_name += ".png"
            plt.savefig(os.path.join(OUTPUT_DIR, save_name))
            plt.clf()

    q_max.append(max(all_q))

plt.bar(range(len(device_set)), q_max,
        color=color, edgecolor='k', zorder=2)

plt.xticks(range(len(device_set)), device_set)
plt.xlabel("Device Number")
plt.ylabel("Highest Q Factor")
plt.grid(axis='y')

plt.tight_layout()
plt.show()
