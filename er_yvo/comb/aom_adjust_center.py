import glob
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# for data
DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Er YVO SHB & AFC"
            "/05_09_24/6Amp/probe/AOMscan/changing_aprobe/0p2_5nW/res")
BG_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Er YVO SHB & AFC"
          "/05_09_24/6Amp/probe/AOMscan/changing_aprobe/0p2_5nW/offres")
TEK_HEADER = ["ParamLabel", "ParamVal", "None", "Seconds", "Volts", "None2"]  # hard-coded from TEK oscilloscope
SCAN_RANGE = 50  # Unit: MHz  # TODO: double check
SCAN_TIME = 0.0128  # Unit: s
GAIN = 1e9  # Unit: V/W
ZERO_FREQ = 196044  # All frequencies are offsets from this (unit: GHz)

EDGE_THRESH = 1  # For finding rising/falling edge of oscilloscope trigger

# for plotting
# plotting parameters
mpl.rcParams.update({'font.size': 12,
                     'figure.figsize': (8, 6)})
OUTPUT_DIR = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/aom_holeburning"
              "/05_09_2024")

PLOT_OD = True  # plot as optical depth

# # plotting output control
PLOT_ALL_SCANS = False  # plot all scans
PLOT_STITCHED = True  # plot all scans for one sequence time together


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
csv_files = glob.glob('*/*/TEK0000.CSV', recursive=True, root_dir=DATA_DIR)
csv_files_freq = glob.glob('*/*/TEK0001.CSV', recursive=True, root_dir=DATA_DIR)
csv_paths = [os.path.join(DATA_DIR, file) for file in csv_files]
csv_paths_freq = [os.path.join(DATA_DIR, file) for file in csv_files_freq]

# read scan center and relative timing
scan_center = np.zeros(len(csv_files))
scan_timing = np.zeros(len(csv_files))
for i, path in enumerate(csv_files):
    path = os.path.normpath(path).split(os.sep)

    # determine timing
    timing_str = path[-2]
    if "beforeburn_at26MHz" in timing_str:
        scan_timing[i] = 1
    elif "afterburn_at26MHz" in timing_str:
        scan_timing[i] = 2
    elif "afterburn_at30MHz" in timing_str:
        scan_timing[i] = 3

    # determine frequency
    freq_str = path[-3]
    freq_str = freq_str[3:]  # remove "f0_"
    freq_str = freq_str[:-3]  # remove "MHz"
    scan_center[i] = int(freq_str)

# read csvs
dfs = [pd.read_csv(path, names=TEK_HEADER) for path in csv_paths]
dfs_freq = [pd.read_csv(path, names=TEK_HEADER) for path in csv_paths_freq]
print(f"Found {len(dfs)} data files.")
print(f"Found {len(dfs_freq)} frequency files.")

# locate background file
bg_file = os.path.join(BG_DIR, "TEK0000.CSV")
bg_file_freq = os.path.join(BG_DIR, "TEK0001.CSV")
print("Reading background file...")
df_bg = pd.read_csv(bg_file, names=TEK_HEADER)
df_bg_freq = pd.read_csv(bg_file_freq, names=TEK_HEADER)


"""
DATA PROCESSING
"""

print("Gathering transmission and background...")

# gather background data
# falling edge case
if df_bg_freq["Volts"].iloc[-1] < df_bg_freq["Volts"][0]:
    scan_edge = [idx for idx in range(1, len(df_bg_freq["Volts"]))
                 if df_bg_freq["Volts"][idx] - df_bg_freq["Volts"][idx-1] < -EDGE_THRESH]
else:
    scan_edge = [idx for idx in range(1, len(df_bg_freq["Volts"]))
                 if df_bg_freq["Volts"][idx] - df_bg_freq["Volts"][idx - 1] > EDGE_THRESH]
if len(scan_edge) > 1:
    warnings.warn("Multiple scan edges found for background, defaulting to first.")
center_idx = scan_edge[0]
time_arr = df_bg_freq["Seconds"]
center_time = time_arr[center_idx]
start_time = np.round(center_time - 0.5*SCAN_TIME, 6)
stop_time = np.round(center_time + 0.5*SCAN_TIME, 6)

start_idx = np.where(time_arr == start_time)[0][0]
stop_idx = np.where(time_arr == stop_time)[0][0]
bg_transmission = df_bg["Volts"][start_idx:stop_idx]
bg_transmission = (bg_transmission / GAIN) * 1e9  # convert to nW

# read starting times, peaks, and single scan
all_scan_midpoints = []  # note: this is the INDEX of the step in the array
all_scan_start = []
all_scan_stop = []
all_scan_transmission = []
all_scan_od = []
all_scan_freq = []

for df, df_freq in zip(dfs, dfs_freq):

    # falling edge case
    if df_freq["Volts"].iloc[-1] < df_freq["Volts"][0]:
        scan_edge = [idx for idx in range(1, len(df_freq["Volts"]))
                     if df_freq["Volts"][idx] - df_freq["Volts"][idx-1] < -EDGE_THRESH]
    else:
        scan_edge = [idx for idx in range(1, len(df_freq["Volts"]))
                     if df_freq["Volts"][idx] - df_freq["Volts"][idx - 1] > EDGE_THRESH]

    # handling of scan center
    if len(scan_edge) == 0:
        warnings.warn("No scan edge found, defaulting to center of window.")
        center_idx = len(df_freq["Volts"]) // 2
    else:
        if len(scan_edge) > 1:
            warnings.warn("Multiple scan edges found, defaulting to first.")
        center_idx = scan_edge[0]
    all_scan_midpoints.append(center_idx)

    time_arr = df_freq["Seconds"]
    center_time = time_arr[center_idx]
    start_time = np.round(center_time - 0.5*SCAN_TIME, 6)
    stop_time = np.round(center_time + 0.5*SCAN_TIME, 6)

    start_idx = np.where(time_arr == start_time)[0][0]
    stop_idx = np.where(time_arr == stop_time)[0][0]
    all_scan_start.append(start_idx)
    all_scan_stop.append(stop_idx)

    transmission = df["Volts"][start_idx:stop_idx]
    transmission = (transmission / GAIN) * 1e9  # convert to nW
    all_scan_transmission.append(transmission)
    freq = np.linspace(-SCAN_RANGE/2, SCAN_RANGE/2, stop_idx-start_idx)
    all_scan_freq.append(freq)

# calculate OD using background (off-res) scan
for trans in all_scan_transmission:
    trans_arr = np.array(trans)
    all_scan_od.append(np.log(bg_transmission / trans_arr))


"""
PLOTTING
"""


# for looking at all scans
if PLOT_ALL_SCANS:
    color = 'tab:blue'

    for i, (trans, freq, od) in enumerate(zip(all_scan_transmission, all_scan_freq, all_scan_od)):
        # plot od
        if PLOT_OD:
            plt.plot(freq, od, color=color)

            plt.xlim((-SCAN_RANGE / 2, SCAN_RANGE / 2))
            plt.title(f"Center {ZERO_FREQ}.{int(scan_center[i])} GHz, sequence {int(scan_timing[i])}")
            plt.xlabel("Detuning (MHz)")
            plt.ylabel("Optical Depth")
            plt.grid(True)

            plt.tight_layout()

            # save fig (avoid error on plotting)
            filename = f"{int(scan_center[i])}_{int(scan_timing[i])}_od.png"
            file_path = os.path.join(OUTPUT_DIR,
                                     "unstitched",
                                     str(int(scan_timing[i])),
                                     filename)
            plt.savefig(file_path)
            plt.clf()

        # plot transmission
        else:
            plt.plot(freq, trans, color=color)

            plt.xlim((-SCAN_RANGE / 2, SCAN_RANGE / 2))
            plt.title(f"Center {ZERO_FREQ}.{int(scan_center[i])} GHz, sequence {int(scan_timing[i])}")
            plt.xlabel("Detuning (MHz)")
            plt.ylabel("Transmission (nW)")
            plt.grid(True)

            plt.tight_layout()

            # save fig (avoid error on plotting)
            filename = f"{int(scan_center[i])}_{int(scan_timing[i])}.png"
            file_path = os.path.join(OUTPUT_DIR,
                                     "unstitched",
                                     str(int(scan_timing[i])),
                                     filename)
            plt.savefig(file_path)
            plt.clf()


if PLOT_STITCHED:
    # plot first sequence
    color = 'tab:blue'

    for i, (trans, freq, od) in enumerate(zip(all_scan_transmission, all_scan_freq, all_scan_od)):
        freq_range_adjusted = (freq + scan_center[i]) / 1e3  # convert to GHz
        if scan_timing[i] == 1:
            if PLOT_OD:
                plt.plot(freq_range_adjusted, od, color)
            else:
                plt.plot(freq_range_adjusted, trans, color)

    plt.title(f"Stitched Data, Before Burning")
    plt.xlabel(f"Detuning from {ZERO_FREQ} (GHz)")
    if PLOT_OD:
        plt.ylabel("Optical Depth")
    else:
        plt.ylabel("Transmission (nW)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # plot second sequence
    color = 'tab:orange'

    for i, (trans, freq, od) in enumerate(zip(all_scan_transmission, all_scan_freq, all_scan_od)):
        freq_range_adjusted = (freq + scan_center[i]) / 1e3  # convert to GHz
        if scan_timing[i] == 2:
            if PLOT_OD:
                plt.plot(freq_range_adjusted, od, color)
            else:
                plt.plot(freq_range_adjusted, trans, color)

    plt.title(f"Stitched Data, After First Burning")
    plt.xlabel(f"Detuning from {ZERO_FREQ} (GHz)")
    if PLOT_OD:
        plt.ylabel("Optical Depth")
    else:
        plt.ylabel("Transmission (nW)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # plot third sequence
    color = 'tab:green'

    for i, (trans, freq, od) in enumerate(zip(all_scan_transmission, all_scan_freq, all_scan_od)):
        freq_range_adjusted = (freq + scan_center[i]) / 1e3  # convert to GHz
        if scan_timing[i] == 3:
            if PLOT_OD:
                plt.plot(freq_range_adjusted, od, color)
            else:
                plt.plot(freq_range_adjusted, trans, color)

    plt.title(f"Stitched Data, After Second Burning")
    plt.xlabel(f"Detuning from {ZERO_FREQ} (GHz)")
    if PLOT_OD:
        plt.ylabel("Optical Depth")
    else:
        plt.ylabel("Transmission (nW)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
