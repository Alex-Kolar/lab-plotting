import glob
import os
import warnings
import numpy as np
import pandas as pd
from lmfit.models import LinearModel, VoigtModel
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle


# for data
DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Er YVO SHB & AFC"
            "/02_07_24/burnprobe/6amp_Bfield/changing_a_pump/0.4 (~1.5uW)/changing_N_pump/probe_AOM_scan_49p5mhz")
BG_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Er YVO SHB & AFC"
          "/02_07_24/burnprobe/6amp_Bfield/changing_a_pump/bg_transmissionlevel_laseroffres")
TEK_HEADER = ["ParamLabel", "ParamVal", "None", "Seconds", "Volts", "None2"]  # hard-coded from TEK oscilloscope
SCAN_RANGE = 49.5  # Unit: MHz
SCAN_TIME = 0.0064  # Unit: s
GAIN = 1e8  # Unit: V/W
PUMP_TIME = 25.6  # Unit: ms (total pump time = N_pump * pump_time)

EDGE_THRESH = 1  # For finding rising/falling edge of oscilloscope trigger

# for fitting
LINEAR_BG_THRESH = 1.5  # All pump times beyond this will use only a linear background (no Voigt)
SAVE_DATA = True  # dump the fitting data to OUTPUT_DIR (for combined plotting later)
SAVE_DATA_FILENAME = "0p4_fits.bin"  # filename for saving detailed above

# for plotting
# plotting parameters
mpl.rcParams.update({'font.size': 12,
                     'figure.figsize': (8, 6)})
OUTPUT_DIR = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/aom_holeburning"
              "/02_07_2024/comb_time_scan/fit_testing")

# # plotting output control
PLOT_ALL_SCANS = False  # plot all scans with fit
PLOT_ALL_HEIGHTS = True  # plot all individually fitted hole heights
PLOT_LINEWIDTHS = True  # plot fitted linewidth of the hole transmission as a function of time
PLOT_BG_LINEWIDTHS = False  # plot fitted linewidth of the background as a function of time
PLOT_BASELINE = True  # plot fitted transmission baseline (background) as a function of time


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

# read timing
pump_times = np.zeros(len(csv_files))
for i, path in enumerate(csv_files):
    path = os.path.normpath(path).split(os.sep)
    pump_str = path[-2]
    pump_str = pump_str.replace('p', '.')
    pump_times[i] = float(pump_str)
pump_times *= (PUMP_TIME / 1e3)  # Unit: s

# sort
csv_paths = [path for _, path in sorted(zip(pump_times, csv_paths))]
csv_paths_freq = [path for _, path in sorted(zip(pump_times, csv_paths_freq))]
pump_times.sort()

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

print("Gathering transmission peaks and background...")

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


# fitting

# do fitting of individual holes
print("Fitting individual holes...")

all_hole_results = []
for i, (freq, od) in enumerate(zip(all_scan_freq, all_scan_od)):
    print(f"\tFitting holes for scan {i+1}/{len(pump_times)}")

    time = pump_times[i]
    if time > LINEAR_BG_THRESH:
        model = LinearModel() - VoigtModel(prefix='hole_')
    else:
        model = VoigtModel(prefix='bg_') + LinearModel() - VoigtModel(prefix='hole_')

    # old guesses
    # hole_sigma_guess = 5
    # hole_amplitude_guess = 20
    # bg_sigma_guess = 20
    # slope_guess = 0.005
    # intercept_guess = 3.5*np.exp(-1.2*i) + 0.8
    # intercept_guess = 3 * np.exp(-1.2 * i) + 0.9
    # intercept_guess = 3 * np.exp(-0.5 * i) - 0.1

    # # for amplitude 1 data on Feb 07 2024
    # hole_sigma_guess = 3
    # hole_amplitude_guess = 20
    # bg_sigma_guess = 20
    # slope_guess = 0.005
    # bg_amplitude_guess = 100
    # if i == 0:
    #     intercept_guess = 3 * np.exp(-0.9 * i) - 0.1
    # elif i == 2:
    #     intercept_guess = 0.5
    #     bg_amplitude_guess = 1
    # else:
    #     if time < LINEAR_BG_THRESH:
    #         intercept_guess = 3 * np.exp(-0.9 * i) - 0.2
    #     else:
    #         intercept_guess = 0.8

    # # for amplitude 0.6 data on Feb 07 2024
    # hole_sigma_guess = 3
    # hole_amplitude_guess = 20
    # bg_sigma_guess = 20
    # slope_guess = 0.005
    # bg_amplitude_guess = 100 * np.exp(-0.9 * i)
    # if i == 1:
    #     intercept_guess = 1.3
    # elif i == 2:
    #     intercept_guess = 1
    # elif i == 3:
    #     intercept_guess = 1
    # elif i == 12:
    #     intercept_guess = 0.5
    # else:
    #     if time > LINEAR_BG_THRESH:
    #         intercept_guess = 0.6
    #     else:
    #         intercept_guess = 3 * np.exp(-0.8 * i) - 0.4

    # for amplitude 0.4 data on Feb 07 2024
    hole_sigma_guess = 4
    hole_amplitude_guess = 5
    if time > LINEAR_BG_THRESH:
        intercept_guess = 0.8
        slope_guess = 0.008
    else:
        if i == 0:
            intercept_guess = 1.5
            slope_guess = 0.02
            bg_sigma_guess = 20
            bg_amplitude_guess = 200
            hole_sigma_guess = 5
            hole_amplitude_guess = 20
        elif i == 3:
            intercept_guess = 1
            slope_guess = 0.008
            bg_sigma_guess = 20
            bg_amplitude_guess = 50
        elif i == 4:
            intercept_guess = 1
            slope_guess = 0.008
            bg_sigma_guess = 20
            bg_amplitude_guess = 50
        else:
            intercept_guess = 1
            slope_guess = 0.008
            bg_sigma_guess = 20
            bg_amplitude_guess = 100 * np.exp(-0.5 * i)

    params = model.make_params()
    params['hole_amplitude'].set(min=0, max=5)
    params['hole_sigma'].set(min=1, max=10)

    if time > LINEAR_BG_THRESH:
        result_hole = model.fit(od, x=freq, params=params,
                                hole_sigma=hole_sigma_guess, hole_amplitude=hole_amplitude_guess,
                                slope=slope_guess, intercept=intercept_guess)
    else:
        params['bg_amplitude'].set(min=0)
        params['bg_sigma'].set(min=10)
        result_hole = model.fit(od, x=freq, params=params,
                                bg_sigma=bg_sigma_guess, hole_sigma=hole_sigma_guess,
                                bg_amplitude=bg_amplitude_guess, hole_amplitude=hole_amplitude_guess,
                                slope=slope_guess, intercept=intercept_guess)

    all_hole_results.append(result_hole)

print("")
print(f"FIT REPORT (T_pump = {pump_times[0]})")
print(all_hole_results[0].fit_report())


"""
SAVING DATA
"""

if SAVE_DATA:
    results_to_save = [res.summary() for res in all_hole_results]
    to_save = (LINEAR_BG_THRESH, pump_times, results_to_save)
    with open(SAVE_DATA_FILENAME, 'wb') as fh:
        pickle.dump(to_save, fh)


"""
PLOTTING
"""


# for looking at all scans
if PLOT_ALL_SCANS:
    color = 'tab:blue'

    for i, (freq, od) in enumerate(zip(all_scan_freq, all_scan_od)):
        time = round(pump_times[i], 3)
        res = all_hole_results[i]

        plt.plot(freq, od, color=color, label="Data")
        plt.plot(freq, res.init_fit, '--r', label="Initial Guess")
        plt.plot(freq, res.best_fit, '--k', label='Fit')

        # construct background only data (no hole)
        slope = res.params["slope"]
        intercept = res.params["intercept"]

        if pump_times[i] > LINEAR_BG_THRESH:
            bg_model = LinearModel()
            y_vals = bg_model.eval(x=freq,
                                   slope=slope, intercept=intercept)
        else:
            amplitude = res.params["bg_amplitude"]
            center = res.params["bg_center"]
            sigma = res.params["bg_sigma"]
            gamma = res.params["bg_gamma"]

            bg_model = LinearModel() + VoigtModel()
            y_vals = bg_model.eval(x=freq,
                                   slope=slope, intercept=intercept,
                                   amplitude=amplitude, center=center,
                                   sigma=sigma, gamma=gamma)

        plt.plot(freq, y_vals, '--g', label="Background Only")

        plt.xlim((-SCAN_RANGE/2, SCAN_RANGE/2))
        plt.title(f"Fitted Hole, T_pump = {time}")
        plt.xlabel("Detuning (MHz)")
        plt.ylabel("Optical Depth")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        # save fig (avoid error on plotting)
        time_str = str(time)
        time_str = time_str.replace(".", "p")
        file_path = os.path.join(OUTPUT_DIR, f"{time_str}.png")
        plt.savefig(file_path)
        plt.clf()
        # plt.show()


# for looking at all fitted heights + fit
if PLOT_ALL_HEIGHTS:
    color = 'tab:purple'
    fig, ax = plt.subplots()

    def get_height(x):
        return x.params['hole_height'].value

    def get_height_err(x):
        return x.params['hole_height'].stderr

    ax.errorbar(pump_times, list(map(get_height, all_hole_results)),
                yerr=list(map(get_height_err, all_hole_results)),
                capsize=10, marker='o', linestyle='', color=color)
    ax.axvline(LINEAR_BG_THRESH, ls='--', color='k')
    ax.set_xscale('log')

    ax.set_title("Hole Height Decay")
    ax.set_xlabel("Pump Time (s)")
    ax.set_ylabel("Hole Height Fit (OD)")
    ax.grid(True)
    ax.set_ylim((0, 1))

    plt.tight_layout()
    plt.show()


# for studying fitted hole width
if PLOT_LINEWIDTHS:
    fig, ax = plt.subplots()

    def get_linewidth(x):
        width = x.params['hole_fwhm'].value  # unit: MHz
        return width

    def get_linewidth_err(x):
        error = x.params['hole_fwhm'].stderr  # unit: MHz
        return error

    ax.errorbar(pump_times,
                list(map(get_linewidth, all_hole_results)),
                yerr=list(map(get_linewidth_err, all_hole_results)),
                capsize=10, marker='o', linestyle='', color='tab:blue')
    ax.axvline(LINEAR_BG_THRESH, ls='--', color='k')
    ax.set_xscale('log')

    ax.set_title("Hole Linewidth (FWHM) versus Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Linewidth (MHz)")
    ax.grid(True)
    ax.set_ylim((0, 15))

    plt.tight_layout()
    plt.show()


# for studying fitted hole width
if PLOT_BG_LINEWIDTHS:
    fig, ax = plt.subplots()

    def get_linewidth(x):
        width = x.params['bg_sigma'].value  # unit: MHz
        return width

    def get_linewidth_err(x):
        error = x.params['bg_sigma'].stderr  # unit: MHz
        return error

    ax.errorbar(pump_times,
                list(map(get_linewidth, all_hole_results)),
                yerr=list(map(get_linewidth_err, all_hole_results)),
                capsize=10, marker='o', linestyle='', color='tab:orange')
    ax.axvline(LINEAR_BG_THRESH, ls='--', color='k')
    ax.set_xscale('log')

    ax.set_title("Background Linewidth (Sigma) versus Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Linewidth (MHz)")
    ax.grid(True)
    ax.set_ylim((0, 20))

    plt.tight_layout()
    plt.show()


# for studying fitted hole baseline
if PLOT_BASELINE:
    fig, ax = plt.subplots()

    def get_bg(x):
        res, time = x
        if time > LINEAR_BG_THRESH:
            return res.params['intercept'].value
        else:
            line = res.params['intercept'].value
            voigt = res.params['bg_height'].value
            return line + voigt

    def get_bg_err(x):
        res, time = x
        if time > LINEAR_BG_THRESH:
            return res.params['intercept'].stderr
        else:
            line_err = res.params['intercept'].stderr
            voigt_err = res.params['bg_height'].stderr
            total = np.sqrt((line_err ** 2) + (voigt_err ** 2))
            return total

    ax.errorbar(pump_times, list(map(get_bg, zip(all_hole_results, pump_times))),
                yerr=list(map(get_bg_err, zip(all_hole_results, pump_times))),
                capsize=10, marker='o', linestyle='', color='tab:orange',
                label='Data')
    ax.axvline(LINEAR_BG_THRESH, ls='--', color='k')
    ax.set_xscale('log')

    ax.set_title("Hole Transmission Background versus Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Background (OD)")
    ax.grid(True)
    ax.set_ylim((0, 4))

    plt.tight_layout()
    plt.show()


# # for studying fitted hole area
# if PLOT_AREA:
#     fig, ax = plt.subplots()
#
#     def get_area(x):
#         return x.params['amplitude'].value
#
#     def get_area_err(x):
#         return x.params['amplitude'].stderr
#
#     ax.errorbar(all_hole_centers, list(map(get_area, all_hole_results)),
#                 yerr=list(map(get_area_err, all_hole_results)),
#                 capsize=10, marker='o', linestyle='', color='tab:red')
#     ax.plot(all_hole_centers, result_fit_area.best_fit,
#             '--k')
#     ax.set_xscale('log')
#
#     if LOG_SCALE:
#         title = "Hole Area (Log Scale) versus Time"
#     else:
#         title = "Hole Area versus Time"
#     ax.set_title(title)
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Area (A.U.)")
#     ax.grid('on')
#
#     plt.tight_layout()
#     plt.show()
