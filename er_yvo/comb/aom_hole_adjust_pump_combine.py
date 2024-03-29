import glob
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle


DATA_DIR = "/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Er YVO SHB & AFC/02_07_24/fit_data"

# for plotting
# plotting parameters
mpl.rcParams.update({'font.size': 12,
                     'figure.figsize': (8, 6)})
OUTPUT_DIR = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/aom_holeburning"
              "/02_07_2024/comb_time_scan/fit_testing")
markers = ['o', '^', 's']

# plotting output control
PLOT_ALL_HEIGHTS = True  # plot all individually fitted hole heights
PLOT_LINEWIDTHS = True  # plot fitted linewidth of the hole transmission as a function of time
PLOT_BASELINE = True  # plot fitted transmission baseline (background) as a function of time


"""
FILE PROCESSING
"""

print("Gathering all fit files...")

# locate all files
fit_files = glob.glob('*.bin', root_dir=DATA_DIR)
fit_paths = [os.path.join(DATA_DIR, file) for file in fit_files]

# read amplitudes
amplitudes = np.zeros(len(fit_files))
for i, path in enumerate(fit_files):
    filename = os.path.splitext(path)[0]
    amp_str = filename.split("_")[0]
    amp_str = amp_str.replace('p', '.')
    amplitudes[i] = float(amp_str)

# sort
fit_paths = [path for _, path in sorted(zip(amplitudes, fit_paths))]
amplitudes.sort()

# read files
thresholds = []
combined_pump_times = []
combined_fit_results = []
for file in fit_paths:
    with open(file, 'rb') as fh:
        linear_thresh, pump_times, results = pickle.load(fh)
    thresholds.append(linear_thresh)
    combined_pump_times.append(pump_times)
    combined_fit_results.append(results)

print(f"Found {len(thresholds)} data files.")


"""
PLOTTING
"""


# for looking at all fitted heights + fit
if PLOT_ALL_HEIGHTS:
    fig, ax = plt.subplots()

    for i, results in enumerate(combined_fit_results):
        heights = []
        height_errs = []
        for res in results:
            params = res['params']
            # find height parameter
            height_param = [p for p in params if p[0] == 'hole_height'][0]
            heights.append(height_param[1])
            height_errs.append(height_param[7])

        pump_times = combined_pump_times[i]
        marker = markers[i]
        amplitude = amplitudes[i]

        ax.errorbar(pump_times, heights,
                    yerr=height_errs,
                    capsize=10, marker=marker, linestyle='',
                    label=f"Pump Amplitude {amplitude}")

    ax.set_xscale('log')
    ax.set_title("Hole Height Decay")
    ax.set_xlabel("Pump Time (s)")
    ax.set_ylabel("Hole Height Fit (OD)")
    ax.grid(True)
    ax.set_ylim((0, 1))
    ax.legend()

    plt.tight_layout()
    plt.show()


# for studying fitted hole width
if PLOT_LINEWIDTHS:
    fig, ax = plt.subplots()

    for i, results in enumerate(combined_fit_results):
        widths = []
        width_errs = []
        for res in results:
            params = res['params']
            # find width parameter
            width_param = [p for p in params if p[0] == 'hole_fwhm'][0]
            widths.append(width_param[1])
            width_errs.append(width_param[7])

        pump_times = combined_pump_times[i]
        marker = markers[i]
        amplitude = amplitudes[i]

        ax.errorbar(pump_times, widths,
                    yerr=width_errs,
                    capsize=10, marker=marker, linestyle='',
                    label=f"Pump Amplitude {amplitude}")

    ax.set_xscale('log')
    ax.set_title("Hole Linewidth (FWHM) versus Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Linewidth (MHz)")
    ax.grid(True)
    ax.set_ylim((0, 15))
    ax.legend()

    plt.tight_layout()
    plt.show()


# for studying fitted hole baseline
if PLOT_BASELINE:
    fig, ax = plt.subplots()

    for i, results in enumerate(combined_fit_results):
        pump_times = combined_pump_times[i]
        thresh = thresholds[i]

        baselines = []
        baseline_errs = []
        for pump_time, res in zip(pump_times, results):
            params = res['params']
            # find intercept parameter
            intercept_param = [p for p in params if p[0] == 'intercept'][0]

            if pump_time > thresh:
                # just use intercept
                baselines.append(intercept_param[1])
                baseline_errs.append(intercept_param[7])
            else:
                # add intercept and bg height
                bg_height_param = [p for p in params if p[0] == 'bg_height'][0]
                baselines.append(intercept_param[1] + bg_height_param[1])
                # calculate combined error
                line_err = intercept_param[7]
                voigt_err = bg_height_param[7]
                total = np.sqrt((line_err ** 2) + (voigt_err ** 2))
                baseline_errs.append(total)

        marker = markers[i]
        amplitude = amplitudes[i]

        ax.errorbar(pump_times, baselines,
                    yerr=baseline_errs,
                    capsize=10, marker=marker, linestyle='',
                    label=f"Pump Amplitude {amplitude}")

    ax.set_xscale('log')
    ax.set_title("Hole Transmission Background versus Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Background (OD)")
    ax.grid(True)
    ax.set_ylim((0, 4))
    ax.legend()

    plt.tight_layout()
    plt.show()
