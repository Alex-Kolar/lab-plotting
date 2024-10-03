import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from lmfit.models import LinearModel, BreitWignerModel
from scipy.signal import find_peaks
import pickle


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/10mK/magnet_scan_10022024")
CSV_PATH = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/10mK/magnet_scan_10022024/resonances_scan_10_02_2024.csv")
OUTPUT_DIR = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
              "/new_mounted/10mK_magnet_scan/10mK_10022024/constant_bg/all_scans")
FREQ_RANGE = (194821.800, 194822.669)

EDGE_THRESH = 2

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
bbox = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor='black')
xlim = (1000, 2500)


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
start_file = main_df['FileNumber'][0]

# set up guesses, etc.
model_kwargs = {
    'p1_amplitude': 0.5,
    'p1_center': 380,
    'p1_sigma': 100,
    'p1_q': 0,
    'p2_amplitude': 0.3,
    'p2_center': 290,
    'p2_sigma': 100,
    'p2_q': 0,
    'intercept': 1.5,
    'slope': 0
}
model = (LinearModel()
         + BreitWignerModel(prefix='p1_')
         + BreitWignerModel(prefix='p2_'))
freq_start = FREQ_RANGE[0]
freq_end = FREQ_RANGE[1]
freq_light = (freq_start * 1e3)


# do fitting of each row
mag_fields = []
qs_1 = []
qs_2 = []
for _, row in main_df.iterrows():
    file_no = int(row['FileNumber'])
    B_field = float(row['MagneticField'])
    mag_fields.append(B_field)
    print(f'Fitting row {file_no-start_file+1}/{num_rows}')

    # get data associated with row
    data_df = data_dfs[file_no]
    scan_all = data_df['CH1']
    transmission_all = data_df['CH3']
    trigger_all = data_df['CH2']

    # find probe region
    scan_start = [idx for idx in range(1, len(trigger_all))
                  if trigger_all[idx] - trigger_all[idx-1] > EDGE_THRESH][0]
    scan_end = [idx for idx in range(1, len(trigger_all))
                if trigger_all[idx] - trigger_all[idx-1] < -EDGE_THRESH][0]
    scan_all = np.array(scan_all[scan_start:scan_end])
    transmission_all = np.array(transmission_all[scan_start:scan_end])

    # find peaks
    maxs = find_peaks(scan_all, distance=40000)[0]
    mins = find_peaks(-scan_all, distance=40000)[0]

    if maxs[-1] < mins[-1]:
        mins = mins[:-1]
    if maxs[0] < mins[0]:
        maxs = maxs[1:]

    # lists to store all results for current magnetic field
    qs_1_temp = []
    qs_2_temp = []
    for i, (start, end) in enumerate(zip(mins, maxs)):
        print(f"\tFitting scan {i+1}/{len(mins)}")
        transmission = transmission_all[start:end]
        freq = np.linspace(0, (freq_end - freq_start) * 1e3,
                           num=(end - start))  # unit: MHz

        res = model.fit(transmission, x=freq,
                        **model_kwargs)

        # do plotting
        plt.plot(freq, transmission,
                 color=color, label='Data')
        plt.plot(freq, res.init_fit,
                 'r--', label='Initial Fit')
        plt.plot(freq, res.best_fit,
                 'k--', label='Fit')

        # extract data
        width_1 = res.params[f'p1_sigma'].value  # unit: MHz
        q_1 = freq_light / width_1
        width_2 = res.params[f'p2_sigma'].value  # unit: MHz
        q_2 = freq_light / width_2
        qs_1_temp.append(q_1)
        qs_2_temp.append(q_2)

        text = rf"$\Gamma_1$: {width_1:.3f} MHz"
        text += "\n"
        text += f"$Q_1$: {q_1:.3}"
        text += "\n"
        text += rf"$\Gamma_2$: {width_2:.3f} MHz"
        text += "\n"
        text += f"$Q_2$: {q_2:.3}"

        plt.gcf().text(0.95, 0.5, text,
                       ha='right', va='center', bbox=bbox)

        plt.title(f"Scan number {i} at {B_field} mT")
        plt.xlabel("Detuning (MHz)")
        plt.ylabel("Transmission (A.U.)")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()

        save_name = f"Scan_{file_no:04d}_{i:02d}.png"
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        plt.savefig(os.path.join(OUTPUT_DIR, save_name))
        plt.clf()

    qs_1.append(qs_1_temp)
    qs_2.append(qs_2_temp)


# save final data
save_data = {
    'Magnetic Field': mag_fields,
    'q_1': qs_1,
    'q_2': qs_2
}
save_name_data = f"res_data.bin"
with open(os.path.join(OUTPUT_DIR, save_name_data), "wb") as f:
    pickle.dump(save_data, f)

# # test plotting
# fig, ax = plt.subplots()
# ax2 = ax.twinx()
# ax.plot(scan_all, 'tab:orange')
# ax2.plot(transmission_all)
#
# ax.scatter(mins, scan_all[mins],
#            marker='x', color='k')
# ax.scatter(maxs, scan_all[maxs],
#            marker='o', color='k')
#
# plt.show()
