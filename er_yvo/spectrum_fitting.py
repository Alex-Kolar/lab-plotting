import glob
import os
import re
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from lmfit.models import ConstantModel, VoigtModel
import matplotlib as mpl
import matplotlib.pyplot as plt


# for data
DATA_DIR = "/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Er YVO Holeburning/"
TEK_HEADER = ["ParamLabel", "ParamVal", "None", "Seconds", "Volts", "None2"]  # hard-coded from TEK oscilloscope
NUM_PEAKS = 2  # number of peaks in the range

# for plotting
VERT_OFFSET = 0.05  # for text, in (normalized) data units
PLOT_ALL_SPECTRA = True

# for output
OUTPUT_DIR = "/Users/alexkolar/PycharmProjects/Plotting/output_figs/hole_decay/spectrum"
SAVE = True  # if True, save to OUTPUT_DIR; otherwise, just show


"""
FILE PROCESSING
"""

print("gathering files")

# locate all files
dirs = glob.glob('**/spectrum/*/', recursive=True, root_dir=DATA_DIR)
# csv_files = glob.glob('**/spectrum/*/TEK0000.CSV', recursive=True, root_dir=DATA_DIR)
# csv_files_freq = glob.glob('**/spectrum/*/CH4.CSV', recursive=True, root_dir=DATA_DIR)
csv_paths = [os.path.join(DATA_DIR, dir, 'TEK0000.CSV') for dir in dirs]
csv_paths_freq = [os.path.join(DATA_DIR, dir, 'CH4.CSV') for dir in dirs]
wavemeter_paths = [os.path.join(DATA_DIR, dir, 'wavemeter.txt') for dir in dirs]

# get currents and scanning ranges (expected)
currents = np.zeros(len(dirs))
ranges = np.zeros(len(dirs))
for i, path in enumerate(dirs):
    path = os.path.normpath(path).split(os.sep)

    current_str = path[-3]
    current_str = current_str[:-3]  # remove 'amp'
    currents[i] = int(current_str)

    scan_str = path[-1]
    scan_str = scan_str[:-8]  # remove 'GHz_scan'
    scan_str = scan_str.replace('p', '.')
    ranges[i] = float(scan_str)

# read csvs
dfs = [pd.read_csv(path, names=TEK_HEADER) for path in csv_paths]
dfs_freq = [pd.read_csv(path, names=TEK_HEADER) for path in csv_paths_freq]


"""
DATA PROCESSING
"""

# get scanning ranges (measured)
scan_ranges = []
for wavemeter_path in wavemeter_paths:
    with open(wavemeter_path) as fh:
        lines = fh.readlines()
    scan_vals = lines[1:4]  # these are the lines of file to get max and min freq from
    individual_diffs = []  # this is the values (to average) of measured diffs for one scan
    for pair in scan_vals:
        pair_strs = pair.split(' ')
        pair_strs = [re.sub('[^0-9.]', '', p) for p in pair_strs if p != '']  # remove empty strings and non-numeric
        diff = float(pair_strs[1]) - float(pair_strs[0])
        individual_diffs.append(diff)
    scan_ranges.append(np.mean(individual_diffs))

# create fitting model
peaks = [VoigtModel(prefix='p{}_'.format(i)) for i in range(NUM_PEAKS)]
model = ConstantModel()
for p in peaks:
    model += p

# fit each scan
all_freqs = []
all_trans = []
all_fits = []
all_peak_guesses = []
for i, (df, df_freq) in enumerate(zip(dfs, dfs_freq)):
    start_idx = df_freq["Volts"].argmin()
    stop_idx = df_freq["Volts"].argmax()
    freq_scan = np.linspace(0, scan_ranges[i], stop_idx-start_idx)
    all_freqs.append(freq_scan)

    spectrum_data = df["Volts"][start_idx:stop_idx].copy()
    spectrum_data.reset_index(drop=True, inplace=True)
    spectrum_data = spectrum_data / spectrum_data.max()
    all_trans.append(spectrum_data)

    spectrum_data_inv = 1 - spectrum_data
    peak_guesses_idx, _ = find_peaks(spectrum_data_inv,
                                     prominence=0.1, distance=100)
    all_peak_guesses.append(peak_guesses_idx)

    res = model.fit(spectrum_data_inv, x=freq_scan,
                    p0_center=freq_scan[peak_guesses_idx[0]],
                    p1_center=freq_scan[peak_guesses_idx[1]],
                    p0_sigma=0.1,
                    p1_sigma=0.1)
    # print(res.fit_report())
    all_fits.append(res)


"""
PLOTTING
"""

if PLOT_ALL_SPECTRA:
    for i in range(len(dfs)):
        fit = all_fits[i]
        split_path = os.path.normpath(dirs[i]).split(os.path.sep)
        date = split_path[0]
        current = split_path[1]
        scan = split_path[3]

        plt.plot(all_freqs[i], all_trans[i],
                 label='Data')
        plt.plot(all_freqs[i], 1-fit.best_fit,
                 '--k',
                 label='Fit')
        # plt.plot(all_freqs[i][all_peak_guesses[i]],
        #          all_trans[i][all_peak_guesses[i]],
        #          'x')

        text_label_0 = r"{:.0f} $\pm$ {:.2f} MHz".format(
            fit.params['p0_fwhm'].value * 1e3,
            fit.params['p0_fwhm'].stderr * 1e3,
        )
        text_label_1 = r"{:.0f} $\pm$ {:.2f} MHz".format(
            fit.params['p1_fwhm'].value * 1e3,
            fit.params['p1_fwhm'].stderr * 1e3,
        )
        plt.text(fit.params['p0_center'].value, 1-fit.params['p0_height'].value-VERT_OFFSET,
                 text_label_0,
                 ha='right')
        plt.text(fit.params['p1_center'].value, 1 - fit.params['p1_height'].value - VERT_OFFSET,
                 text_label_1,
                 ha='left')

        plt.title(f"{date} ({current} B-Field)")
        plt.xlabel("Detuning (GHz)")
        plt.ylabel("Transmission (A.U.)")
        plt.legend()
        plt.grid('on')

        plt.tight_layout()
        if SAVE:
            filename = f"{date}_{current}_{scan}.png"
            plt.savefig(os.path.join(OUTPUT_DIR, filename))
            plt.clf()
        else:
            plt.show()
