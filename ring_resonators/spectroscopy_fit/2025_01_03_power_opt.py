"""Data from January 2 2025

Data format for burning:
'XXDBYYMIN'
where XX is attenuation setting in dB and YY is burn time in minutes.
"""

import glob
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import matplotlib.pyplot as plt


# data files
DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/Bulk_crystal/10mK/01032025")
LASER_OFF = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
             "/Bulk_crystal/10mK/01032025/LASEROFF.csv")
BG_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
           "/Bulk_crystal/10mK/01032025/OFFRES.csv")
AOM_RANGE = (67.855, 92.182)  # unit: MHz

# AFC params
N = 9
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
filenames = glob.glob('*DB*MINSIDE.csv', root_dir=DATA_DIR)
attens = []
burn_times = []
dfs = []
for file in filenames:
    attens.append(int(file[0:2]))
    burn_times.append(int(file[4:6]))
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
        ax.plot(od, color=COLOR)
        ax.set_title(f"AOM Scan OD ({attens[i]} dB, {burn_times[i]} Minutes Burn)")
        ax.set_xlabel("Index")
        ax.set_ylabel("Optical Depth")
        fig.tight_layout()
        fig.show()


# final plotting

# short burn time
idx_to_plot = np.where(burn_times == int(1))[0]
cmap = truncate_colormap(CMAP_SHORT, CMAP_OFFSET, 1)
lines = []
for i in idx_to_plot:
    freq = freqs[i]
    od = ods[i]
    line = np.column_stack((freq, od))
    lines.append(line)
line_coll = LineCollection(lines, cmap=cmap)
line_coll.set_array(attens[idx_to_plot])
line_coll.set_clim(0, 20)

fig, ax = plt.subplots()

im = ax.add_collection(line_coll, autolim=True)
ax.set_xlim((-15, 15))
ax.set_ylim((0, 3))
fig.suptitle("Locked Laser AFC Hole Burning", fontsize=16)
ax.set_title(rf"$N$ = {N}, $\Delta$ = {Delta} MHz, $F$ = {F:0.3f}")
ax.set_xlabel("Detuning (MHz)")
ax.set_ylabel("Optical Depth")

fig.tight_layout()

# add colorbar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label("Attenuation (dB) relative to 394 nW")

fig.show()


# long burn time
idx_to_plot = np.where(burn_times == int(10))[0]
cmap = truncate_colormap(CMAP_LONG, CMAP_OFFSET, 1)
lines = []
for i in idx_to_plot:
    freq = freqs[i]
    od = ods[i]
    line = np.column_stack((freq, od))
    lines.append(line)
line_coll = LineCollection(lines, cmap=cmap)
line_coll.set_array(attens[idx_to_plot])
line_coll.set_clim(0, 20)

fig, ax = plt.subplots()

im = ax.add_collection(line_coll, autolim=True)
ax.set_xlim((-15, 15))
ax.set_ylim((0, 3))
fig.suptitle("Locked Laser AFC Hole Burning", fontsize=16)
ax.set_title(rf"$N$ = {N}, $\Delta$ = {Delta} MHz, $F$ = {F:0.3f}")
ax.set_xlabel("Detuning (MHz)")
ax.set_ylabel("Optical Depth")

fig.tight_layout()

# add colorbar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label("Attenuation (dB) relative to 394 nW")

fig.show()
