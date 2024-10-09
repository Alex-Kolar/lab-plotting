import glob
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import BreitWignerModel, ConstantModel
from scipy.signal import find_peaks
import pickle


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/10mK/10012024")
CSV_PATH = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/10mK/10012024/resonances_10_01_2024.csv")
OUTPUT_DIR = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
              "/new_mounted/10mK_compare/10mK_10012024")

# # fitting params
# SMOOTHING = 100
# PEAK_THRESH = 0.9

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
bbox = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor='black')
xlim = (1000, 2500)

PLOT_ALL_RES = True  # plot and save all intermediate results


# # moving average function for peaks
# def moving_average(a, n=3):
#     ret = np.cumsum(a, dtype=float)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n


# read csv
main_df = pd.read_csv(CSV_PATH)

# find and read oscilloscope files
filenames = glob.glob('SDS*.csv', root_dir=DATA_DIR)
data_dfs = {}
for file in filenames:
    file_str = os.path.splitext(file)[0]
    file_num = int(file_str[3:])
    file_path = os.path.join(DATA_DIR, file)
    data_dfs[file_num] = pd.read_csv(file_path, header=10, skiprows=[11])

# main data processing loop
print('Fitting all resonances.')
num_rows = len(main_df['FileNumber'])
# save specific params for plotting combined results
center_by_device = {}
q_by_device = {}

on_res = []
q_vals = []
widths = []

for scan_no, row in main_df.iterrows():
    # get info
    device = int(row['Device'])
    freq_start = row['Min']  # unit: GHz
    freq_end = row['Max']  # unit: GHz
    file_no = int(row['FileNumber'])
    on_res_val = bool(row['on_res'])
    on_res.append(on_res_val)

    print(f'\tFitting row {file_no}/{num_rows}')
    print(f'\t\tDevice: {device}')
    print(f'\t\tFreq start: {freq_start}')

    # get data associated with row
    data_df = data_dfs[file_no]
    ramp = data_df['CH1'].astype(float)
    transmission = data_df['CH3'].astype(float)

    id_min = np.argmin(ramp)
    id_max = np.argmax(ramp)
    transmission = transmission[id_min:id_max]
    transmission.reset_index(drop=True, inplace=True)
    freq = np.linspace(0, (freq_end - freq_start) * 1e3,
                       num=(id_max - id_min))  # unit: MHz

    min_trans = np.argmin(transmission)
    min_trans_left = min_trans - 200
    peaks_to_keep = [min_trans_left, min_trans]
    print(f"\t\tnumber of peaks: {len(peaks_to_keep)}")

    # do fitting (and determine guesses for fit)
    max_trans = max(transmission)
    model = ConstantModel()
    model_kwargs = {}
    amplitudes = []
    for i, peak_idx in enumerate(peaks_to_keep):
        model += BreitWignerModel(prefix=f'p{i}_')
        amp_guess = (max_trans - transmission[peak_idx]) / 2
        amplitudes.append(amp_guess)
        model_kwargs[f'p{i}_amplitude'] = amp_guess
        model_kwargs[f'p{i}_center'] = freq[peak_idx]
        model_kwargs[f'p{i}_sigma'] = 100
        model_kwargs[f'p{i}_q'] = 0
    model_kwargs['c'] = max_trans - sum(amplitudes)
    res = model.fit(transmission, x=freq,
                    **model_kwargs)

    wl = 299792458 / (freq_start * 1e9)  # unit: m
    wl *= 1e9  # unit: nm

    plt.figure()
    plt.plot(freq, transmission, color=color,
             label='Data')
    plt.plot(freq, res.init_fit, 'r--',
             label='Initial Fit')
    plt.plot(freq, res.best_fit, 'k--',
             label='Fit')

    for i, peak_idx in enumerate(peaks_to_keep):
        plt.axvline(freq[peak_idx], ls=':', color='r')

    # extract data
    all_center_curr = []
    all_q_curr = []
    text = ""
    for i in range(len(peaks_to_keep)):
        width = res.params[f'p{i}_sigma'].value  # unit: MHz
        center = res.params[f'p{i}_center'].value  # unit: MHz
        freq_light = (freq_start * 1e3) + center  # unit: MHz
        q = freq_light / width
        all_center_curr.append(freq_light)
        all_q_curr.append(q)
        q_vals.append(q)
        widths.append(width)

        plt.axvline(center, ls=':', color='k')

        if i != 0:
            text += "\n"
        text += rf"$\Gamma_{i + 1}$: {width:.3f} MHz"
        text += "\n"
        text += f"$Q_{i + 1}$: {q:.3}"

    plt.gcf().text(0.95, 0.5, text,
                   ha='right', va='center', bbox=bbox)

    if device in center_by_device:
        center_by_device[device] += all_center_curr
        q_by_device[device] += all_q_curr
    else:
        center_by_device[device] = all_center_curr
        q_by_device[device] = all_q_curr

    plt.title(f"Number {file_no} {wl:.3f} nm ({freq_start * 1e-3:.3f} THz) scan")
    plt.xlabel("Detuning (MHz)")
    plt.ylabel("Transmission (A.U.)")
    plt.grid(True)
    plt.xlim(xlim)

    # legend magic
    legend = plt.legend()
    legend.get_frame().set(**bbox)

    plt.tight_layout()

    save_name = f"D{device}_{file_no}_{wl:.3f}"
    save_name = save_name.replace(".", "_")
    save_name += ".png"
    output_path = os.path.join(OUTPUT_DIR, str(device))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    plt.savefig(os.path.join(output_path, save_name))
    plt.clf()

# save data for q
save_data = {"q": q_by_device,
             "centers": center_by_device}
save_name_data = f"res_data.bin"
with open(os.path.join(OUTPUT_DIR, save_name_data), "wb") as f:
    pickle.dump(save_data, f)


# do some more plotting
plt.plot(range(num_rows), q_vals)
plt.show()

