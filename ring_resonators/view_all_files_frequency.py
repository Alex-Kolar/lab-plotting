import glob
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/Mounted_device/03182024")
SCAN_RANGE = 30 # unit: GHz

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})


# get scan data
filenames = glob.glob('SDS*.csv', recursive=True, root_dir=DATA_DIR)
paths = [os.path.join(DATA_DIR, file) for file in filenames]
dfs = [pd.read_csv(file, header=11) for file in paths]
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

print(numbers)


# plotting
for num, df in zip(numbers, dfs):
    center_freq = df_freq["Frequency (THz)"][num-1]
    ramp = df['Value'].astype(float)
    transmission = df['Value.1'].astype(float)

    id_min = np.argmin(ramp)
    id_max = np.argmax(ramp)
    ramp = ramp[id_min:id_max]
    transmission = transmission[id_min:id_max]
    freq = np.linspace(0, SCAN_RANGE * 1e3, id_max - id_min)  # unit: MHz

    plt.plot(freq, transmission)

    plt.title(f"{center_freq} THz scan")
    plt.xlabel("Detuning (MHz)")
    plt.ylabel("Transmission (A.U.)")

    # plt.xlim((10000, 13000))

    plt.tight_layout()
    plt.show()
