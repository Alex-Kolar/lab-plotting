import glob
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
import matplotlib.cm as cm
import warnings

# for data
DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Er YVO SHB & AFC"
            "/06_21_24/4K_12Amp/pumpnprobe/tr1_196037p485GHz/scanrange520MHz/changing_a_pump/T_wait_1p536ms")
TEK_HEADER = ["ParamLabel", "ParamVal", "None", "Seconds", "Volts", "None2"]  # hard-coded from TEK oscilloscope
SCAN_RANGE = 520  # Unit: MHz
SCAN_TIME = 1.024e-3  # Unit: s
GAIN = 1e7  # Unit: V/W

EDGE_THRESH = 1  # For finding rising/falling edge of oscilloscope trigger

# for plotting
# plotting parameters
mpl.rcParams.update({'font.size': 12,
                     'figure.figsize': (8, 6)})

# plotting params
CMAP_OFFSET = 0.3
CMAP = cm.Blues
xlim = (-SCAN_RANGE/2, SCAN_RANGE/2)
ylim = (0, 15)

PLOT_OD = False  # plot as optical depth


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Function to plot."""
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


"""
FILE PROCESSING
"""

print("Gathering files...")

# locate all files
csv_files = glob.glob('*/TEK0000.CSV', recursive=True, root_dir=DATA_DIR)
csv_files_freq = glob.glob('*/TEK0001.CSV', recursive=True, root_dir=DATA_DIR)
csv_paths = [os.path.join(DATA_DIR, file) for file in csv_files]
csv_paths_freq = [os.path.join(DATA_DIR, file) for file in csv_files_freq]

# read amplitudes
pump_amps = np.zeros(len(csv_files))
for i, path in enumerate(csv_files):
    path = os.path.normpath(path).split(os.sep)
    pump_str = path[-2]
    pump_str = pump_str.replace('p', '.')
    pump_amp = float(pump_str)
    pump_amps[i] = pump_amp

# sort
csv_paths = [path for _, path in sorted(zip(pump_amps, csv_paths))]
csv_paths_freq = [path for _, path in sorted(zip(pump_amps, csv_paths_freq))]
pump_amps.sort()

# read csvs
dfs = [pd.read_csv(path, names=TEK_HEADER) for path in csv_paths]
dfs_freq = [pd.read_csv(path, names=TEK_HEADER) for path in csv_paths_freq]
print(f"Found {len(dfs)} data files.")
print(f"Found {len(dfs_freq)} frequency files.")


"""
DATA PROCESSING
"""


print("Gathering transmission peaks and background...")

# read starting times, peaks, and single scan
all_scan_midpoints = []  # note: this is the INDEX of the step in the array
all_scan_start = []
all_scan_stop = []
all_scan_transmission = []
all_scan_od = []
all_scan_freq = []
all_max_trans = []
for df, df_freq in zip(dfs, dfs_freq):
    # # falling edge case
    # if df_freq["Volts"].iloc[-1] < df_freq["Volts"][0]:
    #     scan_edge = [idx for idx in range(1, len(df_freq["Volts"]))
    #                  if df_freq["Volts"][idx] - df_freq["Volts"][idx-1] < -EDGE_THRESH]
    # else:
    #     scan_edge = [idx for idx in range(1, len(df_freq["Volts"]))
    #                  if df_freq["Volts"][idx] - df_freq["Volts"][idx - 1] > EDGE_THRESH]
    # if len(scan_edge) > 1:
    #     warnings.warn("Multiple scan edges found, defaulting to first.")
    # center_idx = scan_edge[0]
    # all_scan_midpoints.append(center_idx)
    #
    # time_arr = df_freq["Seconds"]
    # center_time = time_arr[center_idx]
    # start_time = np.round(center_time - 0.5*SCAN_TIME, 6)
    # stop_time = np.round(center_time + 0.5*SCAN_TIME, 6)
    #
    # print(start_time)
    #
    # # # TODO: why is this necessary?
    # # start_time += 0.0000002
    # # stop_time += 0.0000002
    #
    # start_idx = np.where(time_arr == start_time)[0][0]
    # stop_idx = np.where(time_arr == stop_time)[0][0]
    # all_scan_start.append(start_idx)
    # all_scan_stop.append(stop_idx)

    # transmission = df["Volts"][start_idx:stop_idx]
    transmission = df["Volts"]
    transmission = (transmission / GAIN) * 1e9  # convert to nW
    all_scan_transmission.append(transmission)
    freq = np.linspace(-SCAN_RANGE/2, SCAN_RANGE/2, len(transmission))
    all_scan_freq.append(freq)
    all_max_trans.append(max(transmission))

# calculate OD using background (off-res) scan
for max_trans, trans in zip(all_max_trans, all_scan_transmission):
    trans_arr = np.array(trans)
    all_scan_od.append(np.log(max_trans / trans_arr))


"""
PLOTTING
"""


lines = []

if PLOT_OD:
    plot_lines = all_scan_od
else:
    plot_lines = all_scan_transmission

for freq, trans in zip(all_scan_freq, plot_lines):
    line = np.column_stack((freq, trans))
    lines.append(line)

cmap = truncate_colormap(CMAP, CMAP_OFFSET, 1)
line_coll = LineCollection(lines, cmap=cmap)
line_coll.set_array(pump_amps)
line_coll.set_clim(0, max(pump_amps))

fig, ax = plt.subplots(figsize=(6, 4))

im = ax.add_collection(line_coll, autolim=True)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.grid(True)

# labeling
ax.set_xlabel("Detuning (MHz)")
if PLOT_OD:
    ax.set_ylabel("Optical Depth")
else:
    ax.set_ylabel("Transmission (nW)")
ax.set_title(rf"Pump Power Change")

plt.tight_layout()

# add colorbar
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
cb = fig.colorbar(im, cax=cbar_ax)
cb.set_label(r"Pump Amplitude")
# axcb = fig.colorbar(line_coll_low, ax=ax1)
# axcb.set_label("Pump Amplitude")
# axcb = fig.colorbar(line_coll_high, ax=ax2)
# axcb.set_label("Pump Amplitude")

plt.show()
