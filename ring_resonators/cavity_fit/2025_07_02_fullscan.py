import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# data params
DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/300K_no_erbium/2025_07_02/cavity_scan/device_000005")
CSV_PATH = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/300K_no_erbium/2025_07_02/cavity_scan/device_000005/scans_device_05.csv")
FILE_TO_PRINT = 303


# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
bbox = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor='black')


# read csv
main_df = pd.read_csv(CSV_PATH)
num_files = len(main_df['File'])

# data processing and plotting
fig, ax = plt.subplots()
freq_to_save = None
trans_to_save = None
for _, row in main_df.iterrows():
    file_num = row['File'].astype(int)
    min_freq = row['Minimum']
    max_freq = row['Maximum']

    data_path = os.path.join(DATA_DIR, f'data_{file_num:06}.csv')
    data_df = pd.read_csv(data_path)

    ramp = data_df['Ramp Voltage (V)'].astype(float)
    transmission = data_df['Data Voltage (V)'].astype(float)
    id_min = np.argmin(ramp)
    id_max = np.argmax(ramp)
    transmission = transmission[id_min:id_max]
    transmission.reset_index(drop=True, inplace=True)
    freq = np.linspace(min_freq/1e3, max_freq/1e3,
                       num=(id_max - id_min))  # unit: THz

    ax.plot(freq, transmission, color=color)

    # plot this row individually if requested
    if file_num == FILE_TO_PRINT:
        freq_to_save = freq
        trans_to_save = transmission

ax.set_title('Device 5 Resonance Scan')
ax.set_xlabel('Frequency (THz)')
ax.set_ylabel('Transmission (A.U.)')

fig.tight_layout()
fig.show()


fig, ax = plt.subplots()
ax.plot(freq_to_save, trans_to_save, color=color)

fig.tight_layout()
fig.show()
