"""Data from December 9 2024

Sequence of data files:
1. REF: AOM scan taken with laser off-resonant at 1535 nm
2. PREBURN2: AOM scan taken on-resonant
3. AFTERBURN2: AOM scan taken on-resonant after burning with locked laser
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import os


# data files
LASER_OFF = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
             "/Bulk_crystal/10mK/02202025/BACKGROUND.csv")
BG_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
           "/Bulk_crystal/10mK/02202025/AFC/HI/OFFRES.csv")
DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/Bulk_crystal/10mK/02212025")
OUTPUT_DIR = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
              "/bulk_crystal/10mK/hole_burning/AFC/2025_02_21/10min")
AOM_RANGE = (67.855, 92.182)  # unit: MHz
CENTER_FREQS = {5: 194812.766,
                6: 194812.746,
                7: 194812.790}
BURN_TIME = 10  # for gathering files

# AFC params
N = 99
Delta = 1  # unit: MHz
Delta_f = 0.7  # unit: MHz
F = Delta / (Delta - Delta_f)

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica'})
color = 'cornflowerblue'
color_od = 'coral'
SAVEFIG = True


aom_total = AOM_RANGE[1] - AOM_RANGE[0]

# gather ref levels
df_laser_off = pd.read_csv(LASER_OFF, header=10, skiprows=[11])
off_level = np.mean(df_laser_off['CH2'].astype(float).to_numpy())
df_bg = pd.read_csv(BG_DATA, header=10, skiprows=[11])
ramp_ref = df_bg['CH1'].astype(float).to_numpy()
trans_ref = df_bg['CH2'].astype(float).to_numpy()
trans_ref -= off_level

id_min_ref = np.argmin(ramp_ref)
id_max_ref = np.argmax(ramp_ref)

# plot ref level
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(ramp_ref, color='yellow')
ax2.plot(trans_ref, color='magenta')
ax.set_facecolor('black')
ax.set_title("Reference AOM scan")
ax.set_xlabel("Index")
ax.set_ylabel("Ramp (V)")
ax2.set_ylabel("Photodiode response (V)")
fig.tight_layout()
fig.show()

# get intermediate data
files = glob.glob(f"{BURN_TIME}MIN*.csv", root_dir=DATA_DIR)
dfs = {}
freqs = {}
ods = {}
for file in files:
    full_path = os.path.join(DATA_DIR, file)
    name = os.path.splitext(file)[0]
    try:
        number = int(name[-1])
    except ValueError:
        continue
    df = pd.read_csv(full_path, header=10, skiprows=[11])
    dfs[number] = df

    # convert after to transmission and OD
    ramp = df['CH1'].astype(float).to_numpy()
    transmission = df['CH2'].astype(float).to_numpy()
    transmission -= off_level

    # plot of data
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(ramp, color='yellow')
    ax2.plot(transmission, color='magenta')
    ax.set_facecolor('black')
    ax.set_title("AOM scan")
    ax.set_xlabel("Index")
    ax.set_ylabel("Ramp (V)")
    ax2.set_ylabel("Photodiode response (V)")
    fig.tight_layout()
    fig.show()

    id_min = np.argmin(ramp)
    id_max = np.argmax(ramp)
    transmission = transmission[id_min:id_max]

    # normalize
    id_range = id_max - id_min
    transmission = transmission / trans_ref[id_min_ref:(id_min_ref+id_range)]

    # convert time to frequency
    freq = np.linspace(-aom_total/2, aom_total/2, id_max - id_min)  # unit: MHz
    freqs[number] = freq

    # convert to optical depth
    od = np.log(1 / transmission)
    ods[number] = od


# final plotting
for key in dfs.keys():
    freq = freqs[key]
    od = ods[key]
    try:
        center_freq = CENTER_FREQS[key]
    except KeyError:
        print(f"Ignoring scan {key} (no frequency provided)")
        continue

    fig, ax = plt.subplots(figsize=(6, 4))

    plt.plot(freq, od,
             color=color)

    plt.suptitle("Locked Laser AFC Hole Burning", fontsize=16)
    plt.title(rf"$N$ = {N}, $\Delta$ = {Delta} MHz, $F$ = {F:0.3f}")
    plt.xlabel(f"Detuning (MHz) from {center_freq} (GHz)")
    plt.ylabel("Optical Depth")

    plt.xlim((-aom_total/2, aom_total/2))
    # plt.xlim((-2, 2))
    plt.ylim(0, 4)

    plt.tight_layout()
    if SAVEFIG:
        filename = str(center_freq).replace('.', '_')
        plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}_mod.png"))
        plt.clf()
    else:
        plt.show()
