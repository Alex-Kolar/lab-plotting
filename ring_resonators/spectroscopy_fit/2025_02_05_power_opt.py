"""Data from January 2 2025

Data format for burning:
'XXDB_YYMIN_99_SIDE'
where XX is attenuation setting in dB and YY is burn time in minutes.
99 indicates N = 99 for the burned comb, and SIDE indicates sideband burning.
"""

import glob
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.pyplot as plt


# data files
DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/Bulk_crystal/10mK/02052025")
LASER_OFF = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
             "/Bulk_crystal/10mK/02052025/LASEROFF.csv")
BG_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
           "/Bulk_crystal/10mK/02052025/OFFRES.csv")
AOM_RANGE = (67.855, 92.182)  # unit: MHz
# NOTE: need to convert (halve) because different voltage was used
AOM_CENTER = sum(AOM_RANGE)/2
AOM_DIFFS = tuple(f - AOM_CENTER for f in AOM_RANGE)
AOM_RANGE = tuple(AOM_CENTER + d/2 for d in AOM_DIFFS)

# AFC params
N = 99
Delta = 1  # unit: MHz
Delta_f = 0.7  # unit: MHz
F = Delta / (Delta - Delta_f)

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica'})
PLOT_ALL = True
COLOR = 'cornflowerblue'
CMAP_OFFSET = 0.3
CMAP_SHORT = cm.Blues  # for 1 minute burning time
CMAP_LONG = cm.Greens  # for 10 minute burning time


# plotting helper function
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


# gather reference data
df_laser_off = pd.read_csv(LASER_OFF, header=10, skiprows=[11])
off_level = np.mean(df_laser_off['CH2'].astype(float).to_numpy())
df_bg = pd.read_csv(BG_DATA, header=10, skiprows=[11])

# gather ref level
ramp_ref = df_bg['CH1'].astype(float).to_numpy()
trans_ref = df_bg['CH2'].astype(float).to_numpy()
trans_ref -= off_level

id_min_ref = np.argmin(ramp_ref)
id_max_ref = np.argmax(ramp_ref)

# plot ref level
if PLOT_ALL:
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


# gather data for afc burning
filenames = glob.glob('*DB_*MIN_99_SIDE.csv', root_dir=DATA_DIR)
attens = []
burn_times = []
dfs = []
for file in filenames:
    parts = file.split('_')
    attens.append(int(parts[0][:2]))
    burn_times.append(int(parts[1][:2]))
    full_file = os.path.join(DATA_DIR, file)
    dfs.append(pd.read_csv(full_file, header=10, skiprows=[11]))
attens = np.array(attens)
burn_times = np.array(burn_times)


# convert data to transmission and OD
freqs = []
trans = []
ods = []
for i, df in enumerate(dfs):
    ramp = df['CH1'].astype(float).to_numpy()
    transmission = df['CH2'].astype(float).to_numpy()
    transmission -= off_level

    # plot of data
    if PLOT_ALL:
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax.plot(ramp, color='yellow')
        ax2.plot(transmission, color='magenta')
        ax.set_facecolor('black')
        ax.set_title(f"AOM Scan ({attens[i]} dB, {burn_times[i]} Minutes Burn)")
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
    trans.append(transmission)

    # convert time to frequency
    aom_total = AOM_RANGE[1] - AOM_RANGE[0]
    freq = np.linspace(-aom_total/2, aom_total/2, id_max - id_min)  # unit: MHz
    freqs.append(freq)

    # convert to optical depth
    od = np.log(1 / transmission)
    ods.append(od)

    # plot of data
    if PLOT_ALL:
        fig, ax = plt.subplots()
        ax.plot(freq, od, color=COLOR)
        fig.suptitle(f"AOM Scan OD ({attens[i]} dB, {burn_times[i]} Minutes Burn)")
        ax.set_title(rf"$N$ = {N}, $\Delta$ = {Delta} MHz, $F$ = {F:0.3f}")
        ax.set_xlabel("Detuning (MHz)")
        ax.set_ylabel("Optical Depth")
        fig.tight_layout()
        fig.show()

