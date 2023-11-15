import glob
import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from lmfit import Parameters, Model
from lmfit.models import LorentzianModel, ConstantModel, ExponentialModel, LinearModel, VoigtModel, GaussianModel
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# for data
DATA_DIR = "/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Er YVO Holeburning/11_01_23/6amp/hole/Nruns"
TEK_HEADER = ["ParamLabel", "ParamVal", "None", "Seconds", "Volts", "None2"]  # hard-coded from TEK oscilloscope
SCAN_RANGE = 40  # Unit: MHz
REMOVE_LAST = True  # to remove last t_wait segment of data. Only to be used for October 12 data.

# for peak finding
PROMINENCE = 0.01
DISTANCE = 100  # TODO: better way to explicitly calculate this?
PROMINENCE_SCAN = 1

# for fitting of data
LOG_SCALE = True

# for plotting
# plotting parameters
mpl.rcParams.update({'font.size': 12,
                     'figure.figsize': (8, 6)})
xlim_all_plots = (-1, 11)
PLOT_BG = False
PLOT_DECAY = True

# plotting output control
PLOT_ALL_SCANS = False  # plot all scans with background average
PLOT_ALL_PEAKS = True  # plot all peak transmissions, with fitted double-decay exponential
PLOT_ALL_AMPLITUDES = False  # plot all peaks minus minimum of transmission scan, with fitted double-decay
PLOT_ALL_HEIGHTS = False  # plot all individually fitted hole heights, with fitted double-decay
PLOT_STACKED_SCANS = False  # plot all peak transmissions, with color gradient and with no t_wait offset
PLOT_SINGLE_SCAN = False  # plot an individual oscilloscope scan (for troubleshooting)
PLOT_SINGLE_SCAN_HOLES = False  # plot an individual transmission scan, with fitted hole shapes
PLOT_LINEWIDTHS = False  # plot fitted linewidth of the hole transmission as a function of time
PLOT_BASELINE = False  # plot fitted transmission baseline as a function of time
PLOT_AREA = True  # plot fitted area of hole as function of time


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


# fit functions
def decay_double(x, amp_fast, amp_slow, tau_fast, tau_slow, offset):
    return amp_fast * np.exp(-x / tau_fast) + amp_slow * np.exp(-x / tau_slow) + offset


def decay_double_log(x, amp_fast, amp_slow, tau_fast, tau_slow, offset):
    return np.exp(amp_fast * np.exp(-x / tau_fast) + amp_slow * np.exp(-x / tau_slow) + offset)


"""
FILE PROCESSING
"""

print("Gathering files...")

# locate all files
csv_files = glob.glob('*/center.CSV', recursive=True, root_dir=DATA_DIR)
csv_files_freq = glob.glob('*/CH3.CSV', recursive=True, root_dir=DATA_DIR)
csv_paths = [os.path.join(DATA_DIR, file) for file in csv_files]
csv_paths_freq = [os.path.join(DATA_DIR, file) for file in csv_files_freq]

# read wait times
t_wait = np.zeros(len(csv_files))
for i, path in enumerate(csv_files):
    path = os.path.normpath(path).split(os.sep)
    t_wait_str = path[-2]
    t_wait_str = t_wait_str[:-2]  # remove "ms"
    t_wait_str = t_wait_str.replace('p', '.')
    t_wait[i] = float(t_wait_str)

# sort
csv_paths = [path for _, path in sorted(zip(t_wait, csv_paths))]
csv_paths_freq = [path for _, path in sorted(zip(t_wait, csv_paths_freq))]
t_wait.sort()

if REMOVE_LAST:
    csv_paths = csv_paths[:-1]
    csv_paths_freq = csv_paths_freq[:-1]
    t_wait = t_wait[:-1]

# read csvs
dfs = [pd.read_csv(path, names=TEK_HEADER) for path in csv_paths]
dfs_freq = [pd.read_csv(path, names=TEK_HEADER) for path in csv_paths_freq]
print(f"Found {len(dfs)} data files.")
print(f"Found {len(dfs_freq)} frequency files.")

# data for background
if PLOT_BG:
    # bg_path = DATA_DIR + "/0p128ms/bg_offres/center.CSV"
    # bg_path_freq = DATA_DIR + "/0p128ms/bg_offres/CH3.CSV"
    bg_path = DATA_DIR + "/powerlevel/offres_probe.CSV"
    bg_path_freq = DATA_DIR + "/powerlevel/offres_probe_CH3.CSV"
    df_bg = pd.read_csv(bg_path, names=TEK_HEADER)
    df_bg_freq = pd.read_csv(bg_path_freq, names=TEK_HEADER)


"""
DATA PROCESSING
"""

print("Gathering transmission peaks and background...")

# read starting times, peaks, and single scan
all_peaks = []  # NOTE: this is the INDEX of the peak in the array
all_mins = []
all_starts = []  # NOTE: this is also the INDEX of the first scan in the array
all_peak_times = []
all_scan_edges = []  # NOTE: this is the INDEX of the scan edges in the array
for df, df_freq in zip(dfs, dfs_freq):
    # scan_peaks = find_peaks(df_freq["Volts"], prominence=PROMINENCE_SCAN)[0]
    # scan_mins = find_peaks(-df_freq["Volts"], prominence=PROMINENCE_SCAN)[0]
    scan_rise_edge = [idx for idx in range(1, len(df_freq["Volts"]))
                      if df_freq["Volts"][idx] - df_freq["Volts"][idx-1] > 1]
    scan_fall_edge = [idx for idx in range(1, len(df_freq["Volts"]))
                      if df_freq["Volts"][idx] - df_freq["Volts"][idx-1] < -1]
    scan_peaks = [int((idx_rise+idx_fall)/2)
                  for idx_rise, idx_fall in zip(scan_rise_edge, scan_fall_edge)]
    scan_mins = [int((idx_rise + idx_fall) / 2)
                 for idx_rise, idx_fall in zip(scan_rise_edge[1:], scan_fall_edge[:-1])]

    scan_first_peak = scan_peaks[0]
    all_starts.append(scan_first_peak)
    edges = np.concatenate((scan_peaks, scan_mins))
    edges.sort()
    all_scan_edges.append(edges)

    peaks = find_peaks(df["Volts"],
                       prominence=PROMINENCE, distance=DISTANCE)[0]
    peaks = peaks[peaks > scan_first_peak]
    all_peaks.append(peaks)

    time = df["Seconds"]
    peak_times = time[peaks]
    all_peak_times.append(peak_times)

    trans_min = min(df["Volts"][scan_first_peak:])
    all_mins.append(trans_min)

# get background
if PLOT_BG:
    # start_bg = find_peaks(df_bg_freq["Volts"], prominence=PROMINENCE_SCAN)[0][0]
    scan_fall_edge = [idx for idx in range(1, len(df_bg_freq["Volts"]))
                      if df_bg_freq["Volts"][idx] - df_bg_freq["Volts"][idx - 1] < -1]
    start_bg = scan_fall_edge[0]
    end_bg = scan_fall_edge[-1]
    max_bg = max(df_bg["Volts"][start_bg:end_bg])
    min_bg = min(df_bg["Volts"][start_bg:end_bg])

# accumulate all peaks
all_peaks_combine = []
all_amps_combine = []
all_times_combine = []
for i, df in enumerate(dfs):
    start_idx = all_starts[i]
    peak_heights = df["Volts"][all_peaks[i]]
    time = df["Seconds"].copy()
    time += (t_wait[i]/1e3 - time[start_idx])  # add offset
    peak_times = time[all_peaks[i]]

    peak_amps = (peak_heights / all_mins[i]).tolist()
    peak_heights = peak_heights.tolist()
    peak_times = peak_times.tolist()

    all_peaks_combine += peak_heights
    all_amps_combine += peak_amps
    all_times_combine += peak_times


# fitting
print("Fitting hole peak decay...")

model = Model(decay_double_log)
params = Parameters()
params.add('amp_fast', value=0.2, min=0)
params.add('amp_slow', value=0.2, min=0)
params.add('tau_fast', value=0.0005, min=0)
params.add('tau_slow', value=1, min=0)
params.add('offset', value=0)
result = model.fit(all_peaks_combine, params=params, x=all_times_combine)
print("")
print("FIT REPORT (peak height)")
print(result.fit_report())

model_amp = Model(decay_double_log)
params_amp = Parameters()
params_amp.add('amp_fast', value=0.2, min=0)
params_amp.add('amp_slow', value=0.2, min=0)
params_amp.add('tau_fast', value=0.0005, min=0)
params_amp.add('tau_slow', value=10000, min=0)
params_amp.add('offset', value=0)
result_amp = model_amp.fit(all_amps_combine, params=params_amp, x=all_times_combine)
print("")
print("FIT REPORT (peak amplitude)")
print(result_amp.fit_report())


# do fitting of individual holes
print("")
print("Fitting individual holes...")
# model = LorentzianModel() + LinearModel()
model = VoigtModel() + LinearModel()
all_hole_times_2d = []
all_hole_centers_2d = []
all_hole_results_2d = []
for i, df in enumerate(dfs):
    print(f"\tFitting holes for scan {i+1}/{len(t_wait)}")
    hole_times = []
    hole_centers = []
    hole_results = []
    for j, (start_idx, end_idx) in enumerate(zip(all_scan_edges[i][:-1],
                                                 all_scan_edges[i][1:])):
        time = df["Seconds"][start_idx:end_idx]
        hole_times.append(time)

        trans_data = df["Volts"][start_idx:end_idx]
        center_guess = df["Seconds"][all_peaks[i][j]]
        sigma_guess = 0.00005

        params = model.make_params()
        params['sigma'].set(min=sigma_guess)

        if LOG_SCALE:
            result_hole = model.fit(np.log(trans_data), x=time,
                                    center=center_guess, sigma=sigma_guess)
        else:
            result_hole = model.fit(trans_data, x=time,
                                    center=center_guess, sigma=sigma_guess)
        hole_results.append(result_hole)

        hole_centers.append(result_hole.params['center'].value)

        # # convert linewidth to frequency
        # width_time = result_hole.params['fwhm'].value  # unit: seconds
        # error_time = result_hole.params['fwhm'].stderr  # unit: seconds
        # scaling = SCAN_RANGE / (max(time) - min(time))
        # width = width_time * scaling  # unit: MHz
        # try:
        #     error = error_time * scaling  # unit: MHz
        # except TypeError:
        #     print("Failed to get hole params.")
        #     print("Fit report:")
        #     print(result_hole.fit_report())
        #     # raise Exception()
        # linewidths.append(width)
        # errors.append(error)

    all_hole_times_2d.append(hole_times)
    all_hole_centers_2d.append(hole_centers)
    all_hole_results_2d.append(hole_results)

print("")
print("FIT REPORT (first hole fitting)")
print(all_hole_results_2d[-1][0].fit_report())

# LINEWIDTHS_TO_PRINT = -1
# print("")
# print(f"Linewidths (FWHM) for t_wait = {t_wait[LINEWIDTHS_TO_PRINT]}")
# for i, (lw, err) in enumerate(zip(all_hole_linewidth[LINEWIDTHS_TO_PRINT],
#                                   all_hole_error[LINEWIDTHS_TO_PRINT])):
#     print(f"\t{i+1}: {lw} +/- {err} MHz")
#
# HEIGHTS_TO_PRINT = -7
# print("")
# print(f"Fitted heights for t_wait = {t_wait[HEIGHTS_TO_PRINT]}")
# for i, (h, err) in enumerate(zip(all_hole_amplitudes[HEIGHTS_TO_PRINT],
#                                   all_hole_amp_error[HEIGHTS_TO_PRINT])):
#     print(f"\t{i+1}: {h} +/- {err} (A.U.)")

# reshape all hole data
all_hole_times = []
all_hole_results = []
all_hole_centers = []
for i, df in enumerate(dfs):
    start_idx = all_starts[i]
    time_start = df["Seconds"][start_idx]
    # time += (t_wait[i]/1e3 - time[start_idx])  # add offset
    centers = np.array(all_hole_centers_2d[i])
    centers -= time_start
    centers += (t_wait[i] / 1e3)
    centers = centers.tolist()

    all_hole_times += all_hole_times_2d[i]
    all_hole_results += all_hole_results_2d[i]
    all_hole_centers += centers


# fit double exponential of hole height
print("")
print("Fitting hole height...")
model = Model(decay_double)
# model = ExponentialModel() + ConstantModel()
params = Parameters()
params.add('amp_fast', value=0.45, min=0)
params.add('amp_slow', value=0.1, min=0)
params.add('tau_fast', value=0.005, min=0)
params.add('tau_slow', value=10, min=0)
params.add('offset', value=0)
# result_fit_height = model.fit(all_amplitudes_combine, x=all_centers_combine)
print("")
print("FIT REPORT (fitted peak height)")
# print(result_fit_height.fit_report())

# fit exponential decay of background T_0
bg = list(map(lambda x: x.params['intercept'].value, all_hole_results))
print(bg)
print("")
print("Fitting hole background decay...")
model_bg = ExponentialModel() + ConstantModel()
result_bg = model_bg.fit(bg, x=all_hole_centers)
print("")
print("FIT REPORT (background decay)")
print(result_bg.fit_report())


"""
PLOTTING
"""


# for looking at all scans
if PLOT_ALL_SCANS:
    color = 'tab:blue'
    for i, df in enumerate(dfs):
        start_idx = all_starts[i]
        time = df["Seconds"][start_idx:]
        transmission = df["Volts"][start_idx:]

        # time *= 1e3  # convert to ms
        time += (t_wait[i]/1e3 - time[start_idx])  # add offset

        if i == 0:
            plt.plot(time, transmission, label="Transmission", color=color)
        else:
            plt.plot(time, transmission, color=color)

    if PLOT_BG:
        plt.fill_between(xlim_all_plots, max_bg, min_bg, label="Background",
                         color='tab:gray', alpha=0.2)

    plt.xlim(xlim_all_plots)
    plt.title("Hole Transmission Decay (6A B-Field)")
    plt.xlabel("Time (s)")
    plt.ylabel("Transmission (A.U.)")
    plt.legend()
    plt.grid('on')

    plt.tight_layout()
    plt.show()


# for looking at all peaks + fit
if PLOT_ALL_PEAKS:
    fig, ax = plt.subplots()

    # color = 'tab:blue'
    # The color below comes from the B-field scan over all transitions in February
    # see previous plotting scripts for its determination.
    color = (0.0, 0.3544953298505101, 0.14229911572472131)
    ax.loglog(all_times_combine, all_peaks_combine,
                 'o', color=color, label='Data')
    ax.loglog(all_times_combine, result.best_fit,
                 'k--', label='Fit')
    if PLOT_DECAY:
        for i, df in enumerate(dfs):
            start_idx = all_starts[i]
            time = df["Seconds"][start_idx:]
            transmission = df["Volts"][start_idx:]

            # time *= 1e3  # convert to ms
            time += (t_wait[i] / 1e3 - time[start_idx])  # add offset

            if i == 0:
                ax.semilogy(time, transmission, label="Transmission",
                             color=color, alpha=0.2)
            else:
                ax.semilogy(time, transmission,
                             color=color, alpha=0.2)

    if PLOT_BG:
        ax.fill_between(xlim_all_plots, max_bg, min_bg, label="Background",
                         color='tab:gray', alpha=0.2)

    # plt.xlim((-0.1, 0.6))
    ax.set_title("Hole Transmission Decay (6A B-Field)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Transmission (A.U.)")
    ax.legend()
    ax.grid('on')

    fig.tight_layout()
    fig.show()
    # plt.savefig('output_figs/hole_decay/11_01_23/fit_all_decay_loglog.pdf')


# for looking at all amplitudes + fit
if PLOT_ALL_AMPLITUDES:
    color = 'tab:orange'
    plt.loglog(all_times_combine, all_amps_combine,
                 'o', color=color, label='Data')
    plt.loglog(all_times_combine, result_amp.best_fit,
                 'k--', label='Fit')

    # plt.xlim((-0.1, 0.6))
    plt.title("Hole Transmission Amplitude Decay (6A B-Field)")
    plt.xlabel("Time (s)")
    plt.ylabel("Transmission (A.U.)")
    plt.legend()
    plt.grid('on')

    plt.tight_layout()
    plt.show()


# for looking at all fitted heights + fit
if PLOT_ALL_HEIGHTS:
    color = 'tab:purple'
    fig, ax = plt.subplots()

    def get_height(x):
        return x.params['height'].value

    def get_height_err(x):
        return x.params['height'].stderr

    ax.errorbar(all_hole_centers, map(get_height, all_hole_results),
                yerr=map(get_height_err, all_hole_results),
                capsize=10, marker='o', linestyle='', color=color)
    # plt.semilogy(all_centers_combine, result_fit_height.best_fit,
    #              'k--', label='Fit')
    # ax.set_xscale('log')

    ax.set_title("Hole Transmission Decay (6A B-Field)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hole Height Fit (A.U.)")
    ax.grid('on')

    plt.tight_layout()
    plt.show()


# for looking at individual scan decay
if PLOT_STACKED_SCANS:
    lines = []
    for i, df in enumerate(dfs):
        peak_amp = df["Volts"][all_peaks[i]]
        times = all_peak_times[i]
        # plt.semilogy(times, peak_amp, '-o')
        line = np.column_stack((times, peak_amp))
        lines.append(line)

    cmap = truncate_colormap(mpl.cm.Blues, 0.3, 1)
    line_coll = LineCollection(lines, cmap=cmap)
    scale = np.log10(t_wait / 1e3)  # convert to s
    line_coll.set_array(scale)

    fig, ax = plt.subplots()
    ax.add_collection(line_coll, autolim=True)
    ax.autoscale_view()
    ax.set_yscale('log')
    axcb = fig.colorbar(line_coll, ax=ax)
    axcb.set_label(r"$\log_{10}(T_{wait})$ (s)")

    ax.set_title("All Peak Transmission Values (6A B-Field)")
    ax.set_xlabel("Time within scan (s)")
    ax.set_ylabel("Transmission (A.U.)")
    ax.grid('on')

    plt.tight_layout()
    plt.show()


# for studying one scan
if PLOT_SINGLE_SCAN:
    SCAN_TO_PLOT = 0

    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    color1 = 'tab:blue'
    ax.plot(dfs[SCAN_TO_PLOT]["Seconds"], dfs[SCAN_TO_PLOT]["Volts"],
            color=color1)
    ax.plot(dfs[SCAN_TO_PLOT]["Seconds"][all_peaks[SCAN_TO_PLOT]],
            dfs[SCAN_TO_PLOT]["Volts"][all_peaks[SCAN_TO_PLOT]],
            'x', color=color1)

    color2 = 'tab:orange'
    ax2.plot(dfs_freq[SCAN_TO_PLOT]["Seconds"],
             dfs_freq[SCAN_TO_PLOT]["Volts"],
             color=color2)
    ax2.plot(dfs_freq[SCAN_TO_PLOT]["Seconds"][all_scan_edges[SCAN_TO_PLOT]],
             dfs_freq[SCAN_TO_PLOT]["Volts"][all_scan_edges[SCAN_TO_PLOT]],
             'x', color=color2)

    plt.tight_layout()
    plt.show()


# for studying individual hole fits
if PLOT_SINGLE_SCAN_HOLES:
    for i in range(len(dfs)):
        color1 = 'tab:blue'
        if LOG_SCALE:
            plt.plot(dfs[i]["Seconds"][all_starts[i]:],
                     np.log(dfs[i]["Volts"][all_starts[i]:]),
                     color=color1, label='Data')
        else:
            plt.plot(dfs[i]["Seconds"][all_starts[i]:],
                     dfs[i]["Volts"][all_starts[i]:],
                     color=color1, label='Data')

        for j, (time, res) in enumerate(zip(all_hole_times[i],
                                            all_hole_results[i])):
            if j == 0:
                plt.plot(time, res.best_fit,
                         'k--', label='Fit')
            else:
                plt.plot(time, res.best_fit,
                         'k--')

        wait_time = t_wait[i]
        plt.title(rf"Hole fitting ($t_{{wait}}$ = {wait_time} ms)")
        plt.xlabel("Time (s)")
        if LOG_SCALE:
            plt.ylabel("Log(Transmission) (A.U.)")
        else:
            plt.ylabel("Transmission (A.U.)")
        plt.grid("on")
        plt.legend()

        plt.tight_layout()
        plt.show()


# for studying fitted hole width
if PLOT_LINEWIDTHS:
    fig, ax = plt.subplots()

    def get_linewidth(x):
        return x.params['FWHM'].value

    def get_linewidth_err(x):
        return x.params['FWHM'].stderr

    ax.errorbar(all_hole_centers, map(get_linewidth, all_hole_results),
                yerr=map(get_linewidth_err, all_hole_results),
                capsize=10, marker='o', linestyle='', color='tab:blue')
    # ax2.plot(all_centers_combine, all_amplitudes_combine,
    #          marker='o', linestyle='', color='tab:purple')
    ax.set_xscale('log')

    if LOG_SCALE:
        title = "Hole Linewidth (FWHM, Log Scale) versus Time"
    else:
        title = "Hole Linewidth (FWHM) versus Time"
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Linewidth (MHz)")
    ax.grid('on')

    plt.tight_layout()
    plt.show()


# for studying fitted hole baseline
if PLOT_BASELINE:
    fig, ax = plt.subplots()

    def get_bg(x):
        return x.params['intercept'].value

    def get_bg_err(x):
        return x.params['intercept'].stderr

    ax.errorbar(all_hole_centers, map(get_bg, all_hole_results),
                yerr=map(get_bg_err, all_hole_results),
                capsize=10, marker='o', linestyle='', color='tab:orange',
                label='Data')
    ax.plot(all_hole_centers, result_bg.best_fit,
            'k--', label='Fit')
    ax.set_xscale('log')

    ax.set_title("Hole Transmission Background versus Time")
    ax.set_xlabel("Time (s)")
    if LOG_SCALE:
        ax.set_ylabel(r"$\log(T_0)$ (A.U.)")
    else:
        ax.set_ylabel(r"$T_0$ (A.U.)")
    ax.grid('on')
    ax.legend()

    plt.tight_layout()
    plt.show()


# for studying fitted hole area
if PLOT_AREA:
    fig, ax = plt.subplots()

    def get_area(x):
        return x.params['amplitude'].value

    def get_area_err(x):
        return x.params['amplitude'].stderr

    ax.errorbar(all_hole_centers, list(map(get_area, all_hole_results)),
                yerr=list(map(get_area_err, all_hole_results)),
                capsize=10, marker='o', linestyle='', color='tab:red')
    ax.set_xscale('log')

    if LOG_SCALE:
        title = "Hole Area (Log Scale) versus Time"
    else:
        title = "Hole Area versus Time"
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Area (A.U.)")
    ax.grid('on')

    plt.tight_layout()
    plt.show()
