import glob
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import BreitWignerModel, LinearModel


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/Mounted_device/03182024")
OUTPUT_DIR = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
              "/mounted/03182024")
SCAN_RANGE = 30  # unit: GHz

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
PLOT_ALL_RES = True


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


# do data fitting
model = BreitWignerModel() + LinearModel()
all_res = []

for i, df in enumerate(dfs):
    ramp = df['CH1'].astype(float)
    transmission = df['CH2'].astype(float)

    id_min = np.argmin(ramp)
    id_max = np.argmax(ramp)
    transmission = transmission[id_min:id_max]
    freq = np.linspace(-(SCAN_RANGE * 1e3)/2, (SCAN_RANGE * 1e3)/2,
                       id_max - id_min)  # unit: MHz

    out = model.fit(transmission, x=freq,
                    center=15000, amplitude=0.02, sigma=200,
                    slope=0, intercept=max(transmission))

    freq_THz = df_freq["Frequency (THz)"][i]
    freq_light = freq_THz * 1e6  # unit: MHz
    q = freq_light / out.params['sigma'].value

    if PLOT_ALL_RES:
        plt.plot(freq, transmission, color=color)
        plt.plot(freq, out.best_fit, 'k--')

        plt.text(out.params["center"].value + 100,
                 min(transmission),
                 f"Q: {q:.3}")

        plt.title(f"{freq_THz} THz scan")
        plt.xlabel("Detuning (MHz)")
        plt.ylabel("Transmission (A.U.)")
        plt.grid(True)

        plt.tight_layout()

        save_name = f"{i:2}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, save_name))
        plt.clf()
