import glob
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import BreitWignerModel, ConstantModel
import pickle


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/10mK/magnet_scan_10012024")
CSV_PATH = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/10mK/magnet_scan_10012024/resonances_scan_10_01_2024.csv")
OUTPUT_DIR = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
              "/new_mounted/10mK_magnet_scan/10mK_10012024/all_scans")
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

# set up guesses, etc.
model_kwargs = {
    'p1_amplitude': 0.5,
    'p1_center': 530,
    'p1_sigma': 100,
    'p1_q': 0,
    'p2_amplitude': 0.3,
    'p2_center': 450,
    'p2_sigma': 100,
    'p2_q': 0,
    'c': 1.5
}
model = (ConstantModel()
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
    print(f'\tFitting row {file_no-start+1}/{num_rows}')

    # get data associated with row
    data_df = data_dfs[file_no]
    ramp = data_df['CH1'].astype(float)
    transmission = data_df['CH3'].astype(float)

    id_min = np.argmin(ramp)
    id_max = np.argmax(ramp)
    transmission = transmission[id_min:id_max]
    transmission.reset_index(drop=True, inplace=True)
    freq = np.linspace(0, (freq_end-freq_start)*1e3,
                       num=(id_max-id_min))  # unit: MHz

    # do fitting
    model = (ConstantModel()
             + BreitWignerModel(prefix='p1_')
             + BreitWignerModel(prefix='p2_'))
    res = model.fit(transmission, x=freq,
                    p1_amplitude=0.5, p1_center=530, p1_sigma=100, p1_q = 0,
                    p2_amplitude=0.3, p2_center=450, p2_sigma=100, p2_q = 0,
                    c=1.5)

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
    qs_1.append(q_1)
    qs_2.append(q_2)

    text = rf"$\Gamma_1$: {width_1:.3f} MHz"
    text += "\n"
    text += f"$Q_1$: {q_1:.3}"
    text += "\n"
    text += rf"$\Gamma_2$: {width_2:.3f} MHz"
    text += "\n"
    text += f"$Q_2$: {q_2:.3}"

    plt.gcf().text(0.95, 0.5, text,
                   ha='right', va='center', bbox=bbox)

    plt.title(f"Scan number {file_no} ({B_field} mT)")
    plt.xlabel("Detuning (MHz)")
    plt.ylabel("Transmission (A.U.)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    save_name = f"Scan_{file_no:04d}.png"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    plt.savefig(os.path.join(OUTPUT_DIR, save_name))
    plt.clf()

# save final data
save_data = {
    'Magnetic Field': mag_fields,
    'q_1': qs_1,
    'q_2': qs_2
}
save_name_data = f"res_data.bin"
with open(os.path.join(OUTPUT_DIR, save_name_data), "wb") as f:
    pickle.dump(save_data, f)
