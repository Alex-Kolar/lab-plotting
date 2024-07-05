import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
import matplotlib.cm as cm
import warnings


# for data
DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Er YVO SHB & AFC"
            "/01_31_24/burnprobe_fm/trippleburn_persistent_deltaf_0p03/changing_pump_time"
            "/N_pump/probe_scan_250mhz")
TEK_HEADER = ["ParamLabel", "ParamVal", "None", "Seconds", "Volts", "None2"]  # hard-coded from TEK oscilloscope
SCAN_RANGE = 250  # Unit: MHz
SCAN_TIME = 0.0064  # Unit: s
GAIN = 1e8  # Unit: V/W
PUMP_TIME = 27.136  # Unit: ms (total time = N_pump * pump_time)

EDGE_THRESH = 1

# plotting params
CMAP_OFFSET = 0.3
CMAP = cm.Greens
ylim = (0, 45)
PLOT_OD = True  # plot as optical depth
LOG_CMAP = True  # use log scale for colormap

LINES_TO_PLOT = np.array([1, 10, 100], dtype=float)
LINES_TO_PLOT *= (PUMP_TIME / 1e3)  # Unit: s
LINES_TO_PLOT.sort()


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


"""
FILE PROCESSING
"""

print("Gathering files...")

# if ZOOMIN:
#     DATA_DIR = os.path.join(DATA_DIR, "zoomin")
# else:
#     DATA_DIR = os.path.join(DATA_DIR, "zoomout")

# locate all files
csv_files = glob.glob('*/TEK0002.CSV', recursive=True, root_dir=DATA_DIR)
csv_files_freq = glob.glob('*/TEK0003.CSV', recursive=True, root_dir=DATA_DIR)
csv_paths = [os.path.join(DATA_DIR, file) for file in csv_files]
csv_paths_freq = [os.path.join(DATA_DIR, file) for file in csv_files_freq]

print(f"Found {len(csv_paths)} data files.")
print(f"Found {len(csv_paths_freq)} frequency files.")
if len(csv_paths) == 0:
    raise Exception("No valid files found.")
if len(csv_paths) != len(csv_paths_freq):
    raise Exception("Mismatch in number of frequency and transmission channels.")

# read timing
pump_times = np.zeros(len(csv_files))
for i, path in enumerate(csv_files):
    path = os.path.normpath(path).split(os.sep)
    pump_str = path[-2]
    pump_str = pump_str.replace('p', '.')
    pump_times[i] = float(pump_str)
pump_times *= (PUMP_TIME / 1e3)  # Unit: s

if LOG_CMAP:
    zero_idx = np.where(pump_times == 0)[0]
    pump_times = np.delete(pump_times, zero_idx)
    csv_paths = np.delete(csv_paths, zero_idx)
    csv_paths_freq = np.delete(csv_paths_freq, zero_idx)

    pump_times = np.log10(pump_times)

cmap_vmin = min(pump_times)
cmap_vmax = max(pump_times)

# sort
csv_paths = [path for _, path in sorted(zip(pump_times, csv_paths))]
csv_paths_freq = [path for _, path in sorted(zip(pump_times, csv_paths_freq))]
pump_times.sort()

# read csvs
dfs = [pd.read_csv(path, names=TEK_HEADER) for path in csv_paths]
dfs_freq = [pd.read_csv(path, names=TEK_HEADER) for path in csv_paths_freq]


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
max_trans = 0
for df, df_freq in zip(dfs, dfs_freq):
    # falling edge case
    if df_freq["Volts"].iloc[-1] < df_freq["Volts"][0]:
        scan_edge = [idx for idx in range(1, len(df_freq["Volts"]))
                     if df_freq["Volts"][idx] - df_freq["Volts"][idx-1] < -EDGE_THRESH]
    else:
        scan_edge = [idx for idx in range(1, len(df_freq["Volts"]))
                     if df_freq["Volts"][idx] - df_freq["Volts"][idx - 1] > EDGE_THRESH]
    if len(scan_edge) > 1:
        warnings.warn("Multiple scan edges found, defaulting to first.")
    center_idx = scan_edge[0]
    all_scan_midpoints.append(center_idx)

    time_arr = df_freq["Seconds"]
    center_time = time_arr[center_idx]
    start_time = np.round(center_time - 0.5*SCAN_TIME, 6)
    stop_time = np.round(center_time + 0.5 * SCAN_TIME, 6)

    start_idx = np.where(time_arr == start_time)[0][0]
    stop_idx = np.where(time_arr == stop_time)[0][0]
    all_scan_start.append(start_idx)
    all_scan_stop.append(stop_idx)

    transmission = df["Volts"][start_idx:stop_idx]
    transmission = (transmission / GAIN) * 1e9  # convert to nW
    all_scan_transmission.append(transmission)
    freq = np.linspace(-SCAN_RANGE/2, SCAN_RANGE/2, stop_idx-start_idx)
    all_scan_freq.append(freq)

    max_trans = max(max_trans, max(transmission))

for trans in all_scan_transmission:
    trans_arr = np.array(trans)
    all_scan_od.append(np.log(max_trans / trans_arr))


"""
PLOTTING
"""

cmap = truncate_colormap(CMAP, CMAP_OFFSET, 1)

fig, ax = plt.subplots(figsize=(8, 4))

for i, freq in enumerate(all_scan_freq):
    if LOG_CMAP:
        # get color information
        cbar_min = np.log10(LINES_TO_PLOT[0])
        cbar_max = np.log10(LINES_TO_PLOT[-1])

        pump_val = 10 ** pump_times[i]

        if pump_times[i] in np.log10(LINES_TO_PLOT):
            color = cmap((pump_times[i] - cbar_min) / (cbar_max - cbar_min))
            label = rf"$T_{{pump}}$ = {round(pump_val, 3)} s"

            if PLOT_OD:
                ax.plot(freq, all_scan_od[i],
                        color=color, label=label)
            else:
                ax.plot(freq, all_scan_transmission[i],
                        color=color, label=label)

    else:
        if pump_times[i] in LINES_TO_PLOT:
            print("bruh")

ax.set_xlabel("Detuning (MHz)")
if PLOT_OD:
    ax.set_ylabel("Optical Depth")
else:
    ax.set_ylabel("Transmission (nW)")
ax.set_xlim((-SCAN_RANGE/2, SCAN_RANGE/2))
ax.grid(True)
ax.legend()

plt.tight_layout()
# plt.show()

# save as CSV
output_dir = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs"
              "/aom_holeburning/01_31_2024/comb_time_scan")
output_file = os.path.join(output_dir, "waterfall_select_od.svg")
plt.savefig(output_file)
