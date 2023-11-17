import os
import re
import numpy as np
import pandas as pd
from lmfit.models import LinearModel, VoigtModel
import matplotlib as mpl
import matplotlib.pyplot as plt


# for data
DATA_DIR = "/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Er YVO Holeburning" \
           "/11_10_23/6amp/spectrum_polarization"
TEK_HEADER = ["ParamLabel", "ParamVal", "None", "Seconds", "Volts", "None2"]  # hard-coded from TEK oscilloscope

# for fitting (HARD CODED)
pi_center = [3.5, 12]
sigma_center = [7, 9]
fit_range = 2

# for plotting
mpl.rcParams.update({'font.size': 12,
                     'figure.figsize': (8, 6)})
xlim = (2, 13)
data_color = 'silver'
fit_color_pi = 'maroon'
fit_color_sigma_1 = 'lightgreen'
fit_color_sigma_2 = 'darkgreen'
offsets = [2, 1, 0.3]


# get data
csv_pi = os.path.join(DATA_DIR, "pi_3.CSV")
csv_pi_sigma = os.path.join(DATA_DIR, "pitosigma.CSV")
csv_sigma = os.path.join(DATA_DIR, "sigma.CSV")
csv_freq = os.path.join(DATA_DIR, "CH4.CSV")
wavemeter_path = os.path.join(DATA_DIR, "wavemeter.txt")

individual_starts = []  # this is the lower frequency value to average
individual_diffs = []  # this is the values (to average) of measured diffs for one scan
with open(wavemeter_path) as fh:
    lines = fh.readlines()
    scan_vals = lines[1:4]  # these are the lines of file to get max and min freq from
    for pair in scan_vals:
        pair_strs = pair.split(' ')
        pair_strs = [re.sub('[^0-9.]', '', p) for p in pair_strs if p != '']  # remove empty strings and non-numeric
        diff = float(pair_strs[1]) - float(pair_strs[0])
        individual_starts.append(float(pair_strs[0]))
        individual_diffs.append(diff)
scan_start = np.mean(individual_starts)
scan_range = np.mean(individual_diffs)

df_pi = pd.read_csv(csv_pi, names=TEK_HEADER)
df_pi_sigma = pd.read_csv(csv_pi_sigma, names=TEK_HEADER)
df_sigma = pd.read_csv(csv_sigma, names=TEK_HEADER)
df_freq = pd.read_csv(csv_freq, names=TEK_HEADER)

# scan range
start_idx = df_freq["Volts"].argmin()
stop_idx = df_freq["Volts"].argmax()
start_time = df_freq["Seconds"][start_idx]
stop_time = df_freq["Seconds"][stop_idx]


# define frequency scans and truncate
def align(df):
    start_idx = df.index[df["Seconds"] == start_time].tolist()[0]
    stop_idx = df.index[df["Seconds"] == stop_time].tolist()[0]
    scan = df["Volts"][start_idx:stop_idx]
    freq = np.linspace(0, scan_range, stop_idx-start_idx)
    return scan, freq


scan_pi, freq_pi = align(df_pi)
scan_pi_sigma, freq_pi_sigma = align(df_pi_sigma)
scan_sigma, freq_sigma = align(df_sigma)

# normalize
scan_pi /= max(scan_pi)
scan_pi_sigma /= max(scan_pi_sigma)
scan_sigma /= max(scan_sigma)

# fitting of data
model = VoigtModel() + LinearModel()
freq_range_sigma_1 = (sigma_center[0] - fit_range/2, sigma_center[0] + fit_range/2)
freq_range_sigma_2 = (sigma_center[1] - fit_range/2, sigma_center[1] + fit_range/2)
freq_range_pi_1 = (pi_center[0] - fit_range/2, pi_center[0] + fit_range/2)
freq_range_pi_2 = (pi_center[1] - fit_range/2, pi_center[1] + fit_range/2)

results_sigma = []
results_pi = []
freq_truncated_sigma = []
freq_truncated_pi = []

for i, (scan, freq) in enumerate(zip([scan_sigma, scan_pi_sigma, scan_pi],
                                     [freq_sigma, freq_pi_sigma, freq_pi])):
    scan_truncated_sigma_1 = np.array([s for f, s in zip(freq, scan)
                                       if freq_range_sigma_1[0] <= f <= freq_range_sigma_1[1]])
    scan_truncated_sigma_2 = np.array([s for f, s in zip(freq, scan)
                                       if freq_range_sigma_2[0] <= f <= freq_range_sigma_2[1]])
    freq_truncated_sigma_1 = [f for f in freq if freq_range_sigma_1[0] <= f <= freq_range_sigma_1[1]]
    freq_truncated_sigma_2 = [f for f in freq if freq_range_sigma_2[0] <= f <= freq_range_sigma_2[1]]

    res_1 = model.fit(1-scan_truncated_sigma_1, x=freq_truncated_sigma_1,
                      center=sigma_center[0], sigma=0.2)
    res_2 = model.fit(1-scan_truncated_sigma_2, x=freq_truncated_sigma_2,
                      center=sigma_center[1], sigma=0.2)
    results_sigma.append((res_1, res_2))
    freq_truncated_sigma.append((freq_truncated_sigma_1, freq_truncated_sigma_2))

    scan_truncated_pi_1 = np.array([s for f, s in zip(freq, scan)
                                    if freq_range_pi_1[0] <= f <= freq_range_pi_1[1]])
    scan_truncated_pi_2 = np.array([s for f, s in zip(freq, scan)
                                    if freq_range_pi_2[0] <= f <= freq_range_pi_2[1]])
    freq_truncated_pi_1 = [f for f in freq if freq_range_pi_1[0] <= f <= freq_range_pi_1[1]]
    freq_truncated_pi_2 = [f for f in freq if freq_range_pi_2[0] <= f <= freq_range_pi_2[1]]

    res_1 = model.fit(1 - scan_truncated_pi_1, x=freq_truncated_pi_1,
                      center=pi_center[0], sigma=0.01, slope=0, amplitude=0.01)
    res_2 = model.fit(1 - scan_truncated_pi_2, x=freq_truncated_pi_2,
                      center=pi_center[1], sigma=0.01, slope=0, amplitude=0.01)
    results_pi.append((res_1, res_2))
    freq_truncated_pi.append((freq_truncated_pi_1, freq_truncated_pi_2))


# plotting
plt.plot(freq_sigma, scan_sigma + offsets[0],
         color=data_color)
plt.plot(freq_pi_sigma, scan_pi_sigma + offsets[1],
         color=data_color)
plt.plot(freq_pi, scan_pi + offsets[2],
         color=data_color)

for i, (freq_pair, res_pair) in enumerate(zip(freq_truncated_sigma, results_sigma)):
    plt.plot(freq_pair[0], 1 - res_pair[0].best_fit + offsets[i],
             color=fit_color_sigma_1)
    plt.plot(freq_pair[0], 1 - res_pair[0].best_fit + offsets[i],
             color=fit_color_sigma_2, ls='--')
    plt.plot(freq_pair[1], 1 - res_pair[1].best_fit + offsets[i],
             color=fit_color_sigma_1)
    plt.plot(freq_pair[1], 1 - res_pair[1].best_fit + offsets[i],
             color=fit_color_sigma_2, ls='--')

for i, (freq_pair, res_pair) in enumerate(zip(freq_truncated_pi, results_pi)):
    plt.plot(freq_pair[0], 1 - res_pair[0].best_fit + offsets[i],
             color=fit_color_pi)
    plt.plot(freq_pair[1], 1 - res_pair[1].best_fit + offsets[i],
             color=fit_color_pi)

plt.xlim(xlim)
plt.title("Selection Rules")
plt.yticks([1, 2, 3],
           [r'$\pi$', r'$\pi + \sigma$', r'$\sigma$'])
plt.xlabel(f"Detuning from {scan_start:0.0f} (GHz)")

plt.tight_layout()
# plt.show()
plt.savefig('output_figs/hole_decay/spectrum/pol_change.png')
