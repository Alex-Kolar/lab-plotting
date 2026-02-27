"""
NOTE: laser was swept from high to low frequency for this measurement
"""

import os
import pandas as pd
import glob as glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import BreitWignerModel, LinearModel
from scipy.signal import find_peaks
import pickle


DATA_DIR = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
            '/Silicon_test_devices/mk_5/chip_2/2026_02_10/thermal_shift')
input_power = 2.02e3  # unit: uW
output_power = 120
freq_start = 195020.245
freq_end = 195045.228


# find all device data
all_files = glob.glob('*DB.csv', root_dir=DATA_DIR)
all_files = sorted(all_files)

all_attenuations = []
all_transmission = []
all_freqs = []
for file in all_files:
    attenuation = float(file[:2])  # unit: dB
    all_attenuations.append(attenuation)

    file_path = os.path.join(DATA_DIR, file)
    data_df = pd.read_csv(file_path, header=10, skiprows=[11])

    ramp = data_df['CH1'].astype(float)
    transmission = data_df['CH2'].astype(float)
    # transmission -= bg_avg

    id_min = np.argmin(ramp)
    id_max = np.argmax(ramp)
    transmission = transmission[id_max:id_min]
    transmission.reset_index(drop=True, inplace=True)
    transmission /= max(transmission)  # normalize
    freq = np.linspace(freq_end-freq_start, 0,
                       num=(id_min-id_max))  # unit: GHz
    all_freqs.append(freq)
    all_transmission.append(transmission)


for freq, trans in zip(all_freqs, all_transmission):
    plt.plot(freq, trans)

plt.xlim(0, 15)
plt.show()
