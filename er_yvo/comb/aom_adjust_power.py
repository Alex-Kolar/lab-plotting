import glob
import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from lmfit.models import VoigtModel, LinearModel, ConstantModel
import matplotlib.pyplot as plt


# for data
DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Er YVO Holeburning"
            "/01_17_24/burnnprobe/pump/transient/changing_a_pump")
TEK_HEADER = ["ParamLabel", "ParamVal", "None", "Seconds", "Volts", "None2"]  # hard-coded from TEK oscilloscope
SCAN_RANGE = 500  # Unit: MHz
SCAN_TIME = 0.0032  # Unit: s
ZOOMIN = False  # dictates if using "zoomin" or "zoomout" data

# plotting output control
PLOT_ALL_SCANS = True  # plot all scans
OUTPUT_DIR = "/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/aom_holeburning/amplitude_scan"


"""
FILE PROCESSING
"""

print("Gathering files...")

if ZOOMIN:
    DATA_DIR = os.path.join(DATA_DIR, "zoomin")
else:
    DATA_DIR = os.path.join(DATA_DIR, "zoomout")

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
    pump_amps[i] = float(pump_str)

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
all_scan_freq = []
for df, df_freq in zip(dfs, dfs_freq):
    scan_fall_edge = [idx for idx in range(1, len(df_freq["Volts"]))
                      if df_freq["Volts"][idx] - df_freq["Volts"][idx-1] < -1]
    assert len(scan_fall_edge) == 1, "Problem finding single falling edge for frequency channel."
    center_idx = scan_fall_edge[0]
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
    # transmission = (transmission / GAIN) * 1e9  # convert to nW
    all_scan_transmission.append(transmission)
    freq = np.linspace(-SCAN_RANGE/2, SCAN_RANGE/2, stop_idx-start_idx)
    all_scan_freq.append(freq)

# fitting of individual scans
print("")
print("Fitting individual scans...")
all_fits = []
model = ConstantModel() - VoigtModel() + VoigtModel(prefix='hole_')
for i, df in enumerate(dfs):
    # if i != 7:
    #     continue
    print(f"\tFitting for scan {i+1}/{len(pump_amps)}")
    res = model.fit(all_scan_transmission[i], x=all_scan_freq[i],
                    c=1.3, center=10, hole_center=20,
                    amplitude=150, hole_amplitude=4,
                    sigma=40, hole_sigma=4)
    all_fits.append(res)

for i, fit in enumerate(all_fits):
    print(f"{pump_amps[i]}:", fit.params['height'].value, fit.params['hole_height'].value)


"""
PLOTTING
"""

if PLOT_ALL_SCANS:
    color = 'tab:blue'
    for i, df in enumerate(dfs):
        freq = all_scan_freq[i]
        transmission = all_scan_transmission[i]

        plt.plot(freq, transmission,
                 color=color, label="Data")
        plt.plot(freq, all_fits[i].best_fit,
                 'k--', label="Fit")
        plt.plot(freq, all_fits[i].init_fit,
                 'r:', label="Initial Guess")

        plt.title(f"Pump Amplitude {pump_amps[i]}")
        plt.xlabel(f"Detuning (MHz)")
        plt.ylabel("Transmission (A.U.)")
        plt.grid(True)
        plt.legend()

        output_filename = str(pump_amps[i])
        output_filename = output_filename.replace(".", "p")
        output_filename += ".png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.clf()


if PLOT_ALL_SCANS:
    color = 'tab:blue'
    for i, df in enumerate(dfs):
        if i == 0:
            continue
        freq = all_scan_freq[i]
        transmission = all_scan_transmission[i]

        plt.plot(freq, transmission-all_scan_transmission[0],
                 color=color, label="Data - 0 amplitude")

        plt.title(f"Pump Amplitude {pump_amps[i]}")
        plt.xlabel(f"Detuning (MHz)")
        plt.ylabel("Transmission - 0 Transmission (A.U.)")
        plt.grid(True)
        plt.legend()

        output_filename = str(pump_amps[i]) + "_subtract"
        output_filename = output_filename.replace(".", "p")
        output_filename += ".png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.clf()
