import glob
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/Planarized_device/cold_scan_12012023/")
SCAN_RANGE = 30.623  # unit: GHz

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})


filenames = glob.glob('*.csv', recursive=True, root_dir=DATA_DIR)
paths = [os.path.join(DATA_DIR, file) for file in filenames]
dfs = [pd.read_csv(file, header=1) for file in paths]

# get wavelengths
wavelengths = []
for file in filenames:
    wl_str = file[:-4]  # remove '.csv'
    wl_str = wl_str[5:]  # remove 'cold_'
    wl_str = wl_str.replace('p', '.')
    wavelengths.append(float(wl_str))

# plotting
for wl, df in zip(wavelengths, dfs):
    ramp = df['Volt'].astype(float)
    transmission = df['Volt.1'].astype(float)

    id_min = np.argmin(ramp)
    id_max = np.argmax(ramp)
    ramp = ramp[id_min:id_max]
    transmission = transmission[id_min:id_max]
    freq = np.linspace(0, SCAN_RANGE * 1e3, id_max - id_min)  # unit: MHz

    plt.plot(freq, transmission)

    plt.title(f"{wl} nm scan")
    plt.xlabel("Detuning (MHz)")
    plt.ylabel("Transmission (A.U.)")

    # plt.xlim((10000, 13000))

    plt.tight_layout()
    plt.show()
