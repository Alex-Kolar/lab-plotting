import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import BreitWignerModel, ConstantModel
from scipy.signal import find_peaks
import pickle


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/y2o3_data/2025_03_11_d11_full")
CSV_PATH = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/y2o3_data/2025_03_11_d11_full/frequencies.csv")
REF_PATH = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/y2o3_data/2025_03_11_d11_full/CSV_0.csv")
OUTPUT_DIR = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
              "/er_y2o3/03112025/testing")

# fitting params
SMOOTHING = 21
PEAK_THRESH = 0.8

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 10})
color = 'cornflowerblue'

c = 3e8  # units: m/s


# moving average function for peaks
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# read reference data
bg_df = pd.read_csv(REF_PATH, header=10, skiprows=[11])
bg_level = np.mean(bg_df['CH2'])

# read main data
all_freq = np.array([])
all_trans = np.array([])
meta_df = pd.read_csv(CSV_PATH)
for _, row in meta_df.iterrows():
    file_letter = row['File']
    file_path = os.path.join(DATA_DIR, f'CSV{file_letter}.csv')
    data_df = pd.read_csv(file_path, header=10, skiprows=[11])

    # print(f'\tFitting file {file_letter}')

    # get data associated with row
    ramp = data_df['CH1'].astype(float)
    transmission = data_df['CH2'].astype(float)
    transmission -= bg_level

    id_min = np.argmin(ramp)
    id_max = np.argmax(ramp)
    transmission = transmission[id_min:id_max]
    transmission.reset_index(drop=True, inplace=True)
    freq = np.linspace(row['Start'], row['End'],
                       num=(id_max - id_min))  # unit: MHz

    all_freq = np.concatenate((all_freq, freq))
    all_trans = np.concatenate((all_trans, transmission))

    # # find peaks to determine number of fits
    # smoothed_data = moving_average(transmission.to_numpy(),
    #                                n=SMOOTHING)
    # peaks, peak_res = find_peaks(-smoothed_data,
    #                              prominence=0.1, distance=50, width=10)
    # peaks_to_keep = [p for p in peaks
    #                  if transmission[p] < (max(transmission) * PEAK_THRESH)]
    # print(f"\t\tnumber of peaks: {len(peaks_to_keep)}")
    #
    # plt.plot(freq, transmission, color=color)
    # plt.plot(freq[peaks_to_keep], transmission[peaks_to_keep],
    #          ls='', marker='x', color='r')
    # plt.xlabel("Frequency (GHz)")
    # plt.ylabel("Transmission (A.U.)")
    # plt.grid(True)
    #
    # save_name = f"D11_{file_letter}.png"
    # plt.savefig(os.path.join(OUTPUT_DIR, save_name))
    # plt.clf()


# convert freq to wl (in nm)
all_wl = (c / all_freq)


fig, ax = plt.subplots(figsize=(4, 3), dpi=300)

ax.plot(all_wl, all_trans, color=color)

ax.set_title(rf"$Y_2O_3$ Bonded Resonances")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Transmission (A.U.)")

fig.tight_layout()
fig.show()
