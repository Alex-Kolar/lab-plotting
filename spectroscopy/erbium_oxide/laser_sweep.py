import glob
import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import pickle


DATA_DIR = '/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Er2O3'
OUTPUT_NAME = 'processed_data.bin'


def first_and_last_idx(data):
    peak_idxs, _ = find_peaks(data, prominence=1)
    trough_idxs, _ = find_peaks(-data, prominence=1)
    idx_start = trough_idxs[0]
    if peak_idxs[0] < idx_start:
        idx_end = peak_idxs[1]
    else:
        idx_end = peak_idxs[0]

    return idx_start, idx_end


freq_filenames = glob.glob(os.path.join(DATA_DIR, '*.npy'))
freq_filenames = sorted(freq_filenames)
trans_filenames = glob.glob(os.path.join(DATA_DIR, '*.csv'))
trans_filenames = sorted(trans_filenames)
num_freq = len(freq_filenames)
num_trans = len(trans_filenames)
assert num_freq == num_trans

data = {}
for i, (freq_filename, trans_filename) in enumerate(zip(freq_filenames, trans_filenames)):
    print(f'Processing file pair {i+1}/{len(trans_filenames)}')

    # confirm files match and get relevant info from filename
    freq_base = os.path.basename(freq_filename)
    trans_base = os.path.basename(trans_filename)
    assert freq_base[:13] == trans_base[:13]

    freq_parts = freq_base.split('_')
    field = float(freq_parts[0])
    target_freq = float(freq_parts[1])
    if field not in data:
        data[field] = {'frequencies': [], 'voltages': []}

    # load data
    freq_array = np.load(freq_filename)
    trans_df = pd.read_csv(trans_filename, skiprows=8, names=['Time', 'Ramp', 'Photodiode'])

    # get start and end points of ramp and frequency
    ramp_start, ramp_end = first_and_last_idx(trans_df['Ramp'])
    freq_start, freq_end = first_and_last_idx(freq_array)

    # get photodiode voltage and interpolate frequency
    voltage = trans_df['Photodiode'][ramp_start:ramp_end]
    xp_ramp = np.linspace(0, 1, ramp_end-ramp_start)
    xp_freq = np.linspace(0, 1, freq_end-freq_start)
    interp_freq = np.interp(xp_ramp, xp_freq, freq_array[freq_start:freq_end])

    # save
    data[field]['frequencies'].append(interp_freq)
    data[field]['voltages'].append(voltage)

    # plt.plot(interp_freq, voltage)
    # plt.title(f'File {i+1}')
    # plt.show()

print('Saving final data')
output_path = os.path.join(DATA_DIR, OUTPUT_NAME)
with open(output_path, 'wb') as f:
    pickle.dump(data, f)
