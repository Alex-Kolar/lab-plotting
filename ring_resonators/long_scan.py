import glob
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit.models import BreitWignerModel, ConstantModel


DATA_DIR = "/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators" \
           "/Planarized_device/long_scan_11142023"

# data taken for frequency
# F_RANGE = 30.623  # units: GHz
# F_CENTER = [195058, 195083, 195108, 195133, 195158, 195183, 195208, 195233, 195258, 195033, 195008, 194983, 194958, 194933, 194908, 194883, 194858]
F_RANGE = 30.5  # units: GHz
F_CENTER = [195058, 195258, 195283, 195308, 195333, 195358, 195383, 195408, 195433, 195458, 195542, 195692]


# fitting parameters
THRESHOLD = 0.06

# plotting parameters
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
figsize = (18, 6)

# plotting output control
PLOT_COMBINED = True
PLOT_INDIVIDUAL = True


# locate all files
csv_files = glob.glob('*.csv', root_dir=DATA_DIR)
csv_paths = [os.path.join(DATA_DIR, file) for file in csv_files]

# sort all files and get scan number
csv_paths = sorted(csv_paths)
dfs = [pd.read_csv(path, header=1) for path in csv_paths]

# truncate data based on frequency scan
all_freq = []
all_trans = []
for df, center in zip(dfs, F_CENTER):
    scan = df['Volt.1']
    id_min = scan.idxmin()
    id_max = scan.idxmax()
    freq = np.linspace((center-F_RANGE/2),
                       (center+F_RANGE/2),
                       id_max - id_min)
    trans = df['Volt'][id_min:id_max]
    trans.reset_index(drop=True, inplace=True)

    all_freq.append(freq)
    all_trans.append(trans)

# fit some files based on threshold
fit_idx = []
fit_results = []
for i, (trans, freq) in enumerate(zip(all_trans, all_freq)):
    if min(trans) < THRESHOLD:
        center_guess = freq[trans.idxmin()]
        fit_idx.append(i)
        print(f"Fitting for scan {i+1}...")

        # fitting
        model = BreitWignerModel() + ConstantModel()
        out = model.fit(trans, x=freq,
                        center=center_guess, amplitude=0.01)
        # print(out.fit_report())
        fit_results.append(out)

        # print out relevant information
        sigma = out.params['sigma'].value  # unit: GHz
        print("\tSigma:", sigma, "GHz")
        q = out.params['center'].value / sigma
        print("\tQ:", q)


# plotting
if PLOT_COMBINED:
    fig, ax = plt.subplots(figsize=figsize)
    for trans, freq in zip(all_trans, all_freq):
        ax.plot(freq, trans, color=color)

    ax.grid('on')
    ax.set_xlabel("Detuning (GHz)")
    ax.set_ylabel("Transmission")

    for res in fit_results:
        center = res.params['center'].value
        sigma = res.params['sigma'].value
        height = res.params['amplitude'].value
        q = center / sigma
        text_label = f"{q:.0f}"
        # ax.text(fit.params['p0_center'].value, 1 - fit.params['p0_height'].value - VERT_OFFSET,
        #         text_label_0,
        #         ha='center')

    fig.tight_layout()
    fig.show()

if PLOT_INDIVIDUAL:
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
