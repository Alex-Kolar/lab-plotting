import glob
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import BreitWignerModel, LinearModel
import pickle


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/10mK/magnet_scan_10012024")
CSV_PATH = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/10mK/magnet_scan_10012024/resonances_scan_10_01_2024.csv")
OUTPUT_DIR = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
              "/new_mounted/10mK_magnet_scan/10mK_10012024/linear_bg/all_scans")
FREQ_RANGE = (194821.651, 194822.523)  # unit: GHz

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
bbox = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor='black')
xlim = (1000, 2500)

PLOT_ALL_RES = True  # plot and save all intermediate results


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

num_rows = len(main_df['FileNumber'])
start = main_df['FileNumber'][0]
freq_start = FREQ_RANGE[0]
freq_end = FREQ_RANGE[1]


# get 1 scan from each magnetic field
magnetic_fields = main_df['MagneticField'].to_numpy()
unique_fields = np.unique(magnetic_fields)
file_nos = main_df['FileNumber'].to_numpy()

mag_field_points = []
frequency_points = []
transmission_points = []

for field in unique_fields:
    idx = np.where(magnetic_fields == field)[0][0]

    file_no = int(file_nos[idx])
    data_df = data_dfs[file_no]
    ramp = data_df['CH1'].astype(float)
    transmission = data_df['CH3'].astype(float)

    id_min = np.argmin(ramp)
    id_max = np.argmax(ramp)
    transmission = transmission[id_min:id_max]
    transmission.reset_index(drop=True, inplace=True)
    freq = np.linspace(0, (freq_end - freq_start) * 1e3,
                       num=(id_max - id_min))  # unit: MHz

    # normalize
    transmission /= transmission[0]

    frequency_points.append(freq)
    transmission_points.append(transmission)
    mag_field_points.append(field * np.ones_like(transmission))

# flatten data
mag_field_flat = np.concatenate(mag_field_points)
frequency_flat = np.concatenate(frequency_points)
transmission_flat = np.concatenate(transmission_points)


# plotting
plt.tripcolor(frequency_flat, mag_field_flat, transmission_flat,
              cmap='magma_r')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnetic Field (mT)')
plt.colorbar(label='Transmission (A.U.)')

plt.xlim((300, 700))
plt.ylim((700, 1000))

plt.tight_layout()
plt.show()
