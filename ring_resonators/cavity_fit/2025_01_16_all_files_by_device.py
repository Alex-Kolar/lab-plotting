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
            "/New_mounted_device/300K_no_erbium/01162025")
CSV_PATH = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/300K_no_erbium/01162025/resonances_01_16_2025.csv")
OUTPUT_DIR = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
              "/new_mounted_no_erbium/room_temp_cavity/room_temp_01162025/test")

# fitting params
SMOOTHING = 21
PEAK_THRESH = 0.85

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
colormap = mpl.colormaps['Purples']

PLOT_ALL_RES = False  # plot and save all intermediate results


# moving average function for peaks
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


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
contrast_by_device = {}

for _, row in main_df.iterrows():
    # get info
    device = int(row['Device'])
    freq_start = row['Min']  # unit: GHz
    freq_end = row['Max']  # unit: GHz
    file_no = int(row['FileNumber'])

    print(f'\tFitting row {file_no}/{num_rows}')
    print(f'\t\tDevice: {device}')
    print(f'\t\tFreq start: {freq_start}')

    # get data associated with row
    data_df = data_dfs[file_no]
    ramp = data_df['CH1'].astype(float)
    transmission = data_df['CH2'].astype(float)

    id_min = np.argmin(ramp)
    id_max = np.argmax(ramp)
    transmission = transmission[id_min:id_max]
    transmission.reset_index(drop=True, inplace=True)
    freq = np.linspace(0, (freq_end-freq_start)*1e3,
                       num=(id_max-id_min))  # unit: MHz

    # find peaks to determine number of fits
    smoothed_data = moving_average(transmission.to_numpy(),
                                   n=SMOOTHING)
    peaks, peak_res = find_peaks(-smoothed_data,
                                 prominence=0.015, distance=50, width=10)
    peaks_to_keep = [p for p in peaks
                     if transmission[p] < (max(transmission)*PEAK_THRESH)]
    print(f"\t\tnumber of peaks: {len(peaks_to_keep)}")

    # do fitting (and determine guesses for fit)
    max_trans = max(transmission)
    model = ConstantModel()
    model_kwargs = {}
    amplitudes = []
    for i, peak_idx in enumerate(peaks_to_keep):
        model += BreitWignerModel(prefix=f'p{i}_')
        amp_guess = max_trans - transmission[peak_idx]
        amplitudes.append(amp_guess)
        model_kwargs[f'p{i}_amplitude'] = amp_guess
        model_kwargs[f'p{i}_center'] = freq[peak_idx]
        model_kwargs[f'p{i}_sigma'] = 100
        model_kwargs[f'p{i}_q'] = 0
    model_kwargs['c'] = max_trans - sum(amplitudes)
    res = model.fit(transmission, x=freq,
                    **model_kwargs)

    wl = 3e8 / (freq_start * 1e9)  # unit: m
    wl *= 1e9  # unit: nm

    plt.plot(freq, transmission, color=color,
             label='Data')
    plt.plot(freq, res.init_fit, 'r--',
             label='Initial Fit')
    plt.plot(freq, res.best_fit, 'k--',
             label='Fit')

    # # peaks for testing
    # plt.plot(freq[peaks_to_keep], transmission[peaks_to_keep],
    #          'x', color='k')

    # extract data
    print("\t\tCenter Frequencies:")
    all_center_curr = []
    all_q_curr = []
    all_contrast_curr = []
    for i in range(len(peaks_to_keep)):
        width = res.params[f'p{i}_sigma'].value  # unit: MHz
        center = res.params[f'p{i}_center'].value  # unit: MHz
        amplitude = res.params[f'p{i}_amplitude'].value
        constant = res.params[f'c'].value

        # get q
        freq_light = (freq_start * 1e3) + center  # unit: MHz
        q = freq_light / width

        # calculate contrast
        contrast = amplitude / (amplitude + constant)

        # save data
        all_center_curr.append(freq_light)
        all_q_curr.append(q)
        all_contrast_curr.append(contrast)

        text = rf"$\Gamma$: {width:.3f} MHz"
        text += "\n"
        text += f"Q: {q:.3}"
        plt.text(center + 500,
                 transmission[peaks_to_keep[i]] - 0.1,
                 text)

        print(f"\t\t\t{freq_light * 1e-3:.3f} GHz ({(3e8 / (freq_light * 1e6)) * 1e9:.3f} nm)")

    if device in center_by_device:
        center_by_device[device] += all_center_curr
        q_by_device[device] += all_q_curr
        contrast_by_device[device] += all_contrast_curr
    else:
        center_by_device[device] = all_center_curr
        q_by_device[device] = all_q_curr
        contrast_by_device[device] = all_contrast_curr


    plt.title(f"Device {device} {wl:.3f} nm ({freq_start * 1e-3:.3f} THz) scan")
    plt.legend(shadow=True)
    plt.xlabel("Detuning (MHz)")
    plt.ylabel("Transmission (A.U.)")
    plt.grid(True)

    plt.tight_layout()

    save_name = f"D{device}_{wl:.3f}"
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


# make bar graph of highest q
bar_x = range(len(q_by_device.keys()))
labels = list(q_by_device.keys())
bar_y = [max(vals) for vals in q_by_device.values()]
plt.bar(bar_x, bar_y,
        color=color, edgecolor='k', zorder=2)

plt.xticks(bar_x, labels)
plt.xlabel("Device Number")
plt.ylabel("Highest Q Factor")
plt.grid(axis='y')

plt.tight_layout()
plt.show()


# make plot of q versus freq for each device
# coloring is based on coupling
for device in center_by_device:
    centers = np.array(center_by_device[device])
    centers *= 1e-6  # convert to THz
    qs = np.array(q_by_device[device])
    contrasts = np.array(contrast_by_device[device])

    # colored stem plot
    # collect lines
    for center, q, contrast in zip(centers, qs, contrasts):
        color_samp = colormap(contrast)
        plt.plot([center, center], [0, q], color=color_samp)
    # do scatter part
    plt.scatter(centers, qs, c=contrasts,
                cmap=colormap, vmin=0, vmax=1, zorder=3)
    plt.axhline(y=0, color='k')
    plt.title(f"Device {device}")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Q Factor")
    plt.colorbar(label="Contrast")

    plt.tight_layout()
    plt.show()
