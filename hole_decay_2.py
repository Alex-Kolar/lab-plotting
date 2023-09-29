import glob
import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from lmfit import Parameters, Model
from lmfit.models import LorentzianModel, ConstantModel
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# for data
DATA_DIR = "/Users/alexkolar/Desktop/Projects/AFC/09_18_23/6Amp/hole"
TEK_HEADER = ["ParamLabel", "ParamVal", "None", "Seconds", "Volts", "None2"]  # hard-coded from TEK oscilloscope
SCAN_RANGE = 20  # Unit: MHz

# for peak finding/fitting
PROMINENCE = 0.01
DISTANCE = 100  # TODO: better way to explicitly calculate this?
PROMINENCE_SCAN = 1
LOG_SCALE = False

# for plotting
# plotting parameters
mpl.rcParams.update({'font.size': 12,
                     'figure.figsize': (8, 6)})
xlim_all_plots = (-1, 11)
PLOT_BG = True
PLOT_DECAY = True

# plotting output control
PLOT_ALL_SCANS = False
PLOT_ALL_PEAKS = False
PLOT_ALL_AMPLITUDES = False
PLOT_STACKED_SCANS = False
PLOT_SINGLE_SCAN = False
PLOT_SINGLE_SCAN_HOLES = True


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


# locate all files
csv_files = glob.glob('*/center.CSV', recursive=True, root_dir=DATA_DIR)
csv_files_freq = glob.glob('*/CH3.CSV', recursive=True, root_dir=DATA_DIR)
csv_paths = [os.path.join(DATA_DIR, file) for file in csv_files]
csv_paths_freq = [os.path.join(DATA_DIR, file) for file in csv_files_freq]

bg_path = DATA_DIR + "/0p128ms/bg_offres/center.CSV"
bg_path_freq = DATA_DIR + "/0p128ms/bg_offres/CH3.CSV"

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

# read csvs
dfs = [pd.read_csv(path, names=TEK_HEADER) for path in csv_paths]
dfs_freq = [pd.read_csv(path, names=TEK_HEADER) for path in csv_paths_freq]
df_bg = pd.read_csv(bg_path, names=TEK_HEADER)
df_bg_freq = pd.read_csv(bg_path_freq, names=TEK_HEADER)


"""
DATA PROCESSING
"""


# read starting times, peaks, and single scan
all_peaks = []  # NOTE: this is the INDEX of the peak in the array
all_mins = []
all_starts = []  # NOTE: this is also the INDEX of the first scan in the array
all_peak_times = []
all_scan_edges = []  # NOTE: this is the INDEX of the scan edges in the array
for df, df_freq in zip(dfs, dfs_freq):
    scan_peaks = find_peaks(df_freq["Volts"], prominence=PROMINENCE_SCAN)[0]
    scan_mins = find_peaks(-df_freq["Volts"], prominence=PROMINENCE_SCAN)[0]
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

start_bg = find_peaks(df_bg_freq["Volts"], prominence=PROMINENCE_SCAN)[0][0]

# get background
max_bg = max(df_bg["Volts"][start_bg:])
min_bg = min(df_bg["Volts"][start_bg:])

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
model = Model(decay_double_log)
params = Parameters()
params.add('amp_fast', value=0.2, min=0)
params.add('amp_slow', value=0.2, min=0)
params.add('tau_fast', value=0.0005, min=0)
params.add('tau_slow', value=1, min=0)
params.add('offset', value=0)
result = model.fit(all_peaks_combine, params=params, x=all_times_combine)
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
print(t_wait)
SCAN_TO_FIT = 4
model = LorentzianModel() + ConstantModel()
all_hole_times = []
all_hole_results = []
for i, (start_idx, end_idx) in enumerate(zip(all_scan_edges[SCAN_TO_FIT][:-1],
                                             all_scan_edges[SCAN_TO_FIT][1:])):
    time = dfs[SCAN_TO_FIT]["Seconds"][start_idx:end_idx]
    all_hole_times.append(time)

    trans_data = dfs[SCAN_TO_FIT]["Volts"][start_idx:end_idx]
    center_guess = dfs[SCAN_TO_FIT]["Seconds"][all_peaks[SCAN_TO_FIT][i]]
    sigma_guess = 0.0005
    if LOG_SCALE:
        result_hole = model.fit(np.log(trans_data), x=time,
                                center=center_guess, sigma=0.0005)
    else:
        result_hole = model.fit(trans_data, x=time,
                                center=center_guess, sigma=0.0005)
    all_hole_results.append(result_hole)

print("")
print("FIT REPORT (first hole fitting)")
print(all_hole_results[0].fit_report())

# convert linewidths to frequency
linewidths = []
errors = []
for time, res in zip(all_hole_times, all_hole_results):
    width_time = res.params['fwhm'].value  # unit: seconds
    error_time = res.params['fwhm'].stderr  # unit: seconds
    scaling = SCAN_RANGE / (max(time) - min(time))
    width = width_time * scaling  # unit: MHz
    error = error_time * scaling  # unit: MHz
    linewidths.append(width)
    errors.append(error)

print("")
print("LINEWIDTHS (FWHM):")
for i, (lw, err) in enumerate(zip(linewidths, errors)):
    print(f"\t{i+1}: {lw:.3f} +/- {err:.3f} MHz")

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


# # for looking at all peaks + fit
if PLOT_ALL_PEAKS:
    color = 'tab:blue'
    plt.semilogy(all_times_combine, all_peaks_combine,
                 'o', label='Data')
    plt.semilogy(all_times_combine, result.best_fit,
                 'k--', label='Fit')
    if PLOT_DECAY:
        for i, df in enumerate(dfs):
            start_idx = all_starts[i]
            time = df["Seconds"][start_idx:]
            transmission = df["Volts"][start_idx:]

            # time *= 1e3  # convert to ms
            time += (t_wait[i] / 1e3 - time[start_idx])  # add offset

            if i == 0:
                plt.loglog(time, transmission, label="Transmission",
                             color=color, alpha=0.2)
            else:
                plt.loglog(time, transmission,
                             color=color, alpha=0.2)

    # plt.xlim((-0.1, 0.6))
    plt.title("Hole Transmission Decay (6A B-Field)")
    plt.xlabel("Time (s)")
    plt.ylabel("Transmission (A.U.)")
    plt.legend()
    plt.grid('on')

    plt.tight_layout()
    plt.show()


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
    SCAN_TO_PLOT = -1

    fig, ax = plt.subplots()
    ax2 = ax.twinx()

    color1 = 'tab:blue'
    ax.plot(dfs[SCAN_TO_PLOT]["Seconds"], dfs[SCAN_TO_PLOT]["Volts"],
            color=color1)
    ax.plot(dfs[SCAN_TO_PLOT]["Seconds"][all_peaks[SCAN_TO_PLOT]],
            dfs[SCAN_TO_PLOT]["Volts"][all_peaks[SCAN_TO_PLOT]],
            'o', color=color1)

    color2 = 'tab:orange'
    ax2.plot(dfs_freq[SCAN_TO_PLOT]["Seconds"],
             dfs_freq[SCAN_TO_PLOT]["Volts"],
             color=color2)
    ax2.plot(dfs_freq[SCAN_TO_PLOT]["Seconds"][all_scan_edges[SCAN_TO_PLOT]],
             dfs_freq[SCAN_TO_PLOT]["Volts"][all_scan_edges[SCAN_TO_PLOT]],
             'o', color=color2)

    plt.tight_layout()
    plt.show()


# for studying individual hole fits
if PLOT_SINGLE_SCAN_HOLES:
    SCAN_TO_PLOT = 4
    assert SCAN_TO_PLOT == SCAN_TO_FIT  # (remove later)
    color1 = 'tab:blue'
    if LOG_SCALE:
        plt.plot(dfs[SCAN_TO_PLOT]["Seconds"][all_starts[SCAN_TO_PLOT]:],
                 np.log(dfs[SCAN_TO_PLOT]["Volts"][all_starts[SCAN_TO_PLOT]:]),
                 color=color1, label='Data')
    else:
        plt.plot(dfs[SCAN_TO_PLOT]["Seconds"][all_starts[SCAN_TO_PLOT]:],
                 dfs[SCAN_TO_PLOT]["Volts"][all_starts[SCAN_TO_PLOT]:],
                 color=color1, label='Data')

    for i, (time, res) in enumerate(zip(all_hole_times, all_hole_results)):
        if i == 0:
            plt.plot(time, res.best_fit,
                     'k--', label='Fit')
        else:
            plt.plot(time, res.best_fit,
                     'k--')

    plt.title(rf"Hole fitting ($t_{{wait}}$ = {t_wait[SCAN_TO_PLOT]} ms)")
    plt.xlabel("Time (s)")
    if LOG_SCALE:
        plt.ylabel("Log(Transmission) (A.U.)")
    else:
        plt.ylabel("Transmission (A.U.)")
    plt.grid("on")
    plt.legend()

    plt.tight_layout()
    plt.show()
