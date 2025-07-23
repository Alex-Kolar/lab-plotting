import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import BreitWignerModel, LinearModel
from scipy.signal import find_peaks
import pickle

from cavity_metrics import calculate_enhancement, calculate_rates


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/Mounted_device_mk_3/4K/2025_07_22/cavity_scan")
CSV_PATH = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/Mounted_device_mk_3/4K/2025_07_22/cavity_scan/resonances_07_22_2025.csv")
LASER_OFF_PATH = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
                  "/Mounted_device_mk_3/4K/2025_07_22/background/data_000000.csv")
OUTPUT_DIR = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
              "/mounted_mk_3/4K_cavity/4K_07222025")
OUTPUT_DIR_ALL = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
                  "/mounted_mk_3/4K_cavity/all_devices")

# fitting params
SMOOTHING = 21
PEAK_THRESH = 0.7

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
color_smooth = 'coral'
bbox = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor='black')
color_factor = 'coral'
colormap = mpl.colormaps['Purples']
color_rate = 'black'

PLOT_ALL_RES = False  # plot and save all intermediate results
PLOT_ALL_RES_TESTING = False  # plot and save all intermediate results with annotated fit information

SAVE_DEVICE_PLOTS = False  # if true, save combined output plots. Otherwise, display with plt.show()
PLOT_FIELD_ENHANCE = False  # plot field enhancement factor for each device
PLOT_RATE = False  # plot rate of pair generation for each device


# moving average function for peaks
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# read csv for device info
main_df = pd.read_csv(CSV_PATH)

# read background data
bg_df = pd.read_csv(LASER_OFF_PATH)
transmission = bg_df['Data Voltage (V)'].astype(float)
bg_avg = np.min(transmission)

# main data processing loop
print('Fitting all resonances.')
num_rows = len(main_df['FileNumber'])
# save specific params for plotting combined results
center_by_device = {}
q_by_device = {}
contrast_by_device = {}

for row_idx, row in main_df.iterrows():
    # get info
    device = int(row['Device'])
    freq_start = row['Min']  # unit: GHz
    freq_end = row['Max']  # unit: GHz
    file_no = int(row['FileNumber'])

    print(f'\tFitting row {row_idx+1}/{num_rows}')
    print(f'\t\tDevice: {device}')
    print(f'\t\tFreq start: {freq_start}')

    # get oscilloscope data
    file_path = os.path.join(DATA_DIR, f'device_{device:02d}', f'data_{file_no:06d}.csv')
    data_df = pd.read_csv(file_path)

    ramp = data_df['Ramp Voltage (V)'].astype(float)
    transmission = data_df['Data Voltage (V)'].astype(float)
    transmission -= bg_avg

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
                                 prominence=0.015, distance=20, width=10)
    peaks_to_keep = [p for p in peaks
                     if transmission[p] < (max(transmission)*PEAK_THRESH)]
    print(f"\t\tnumber of peaks: {len(peaks_to_keep)}")

    # do fitting (and determine guesses for fit)
    max_trans = max(transmission)
    model = LinearModel()
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
    model_kwargs['intercept'] = max_trans - sum(amplitudes)
    model_kwargs['slope'] = 0
    res = model.fit(transmission, x=freq,
                    **model_kwargs)

    # extract data
    total_amp = sum([res.params[f'p{i}_amplitude'].value for i in range(len(peaks_to_keep))])
    wl = 299792458 / (freq_start * 1e9)  # unit: m
    wl *= 1e9  # unit: nm

    all_center_curr = []
    all_q_curr = []
    all_contrast_curr = []
    text = ""
    for i in range(len(peaks_to_keep)):
        width = res.params[f'p{i}_sigma'].value  # unit: MHz
        center = res.params[f'p{i}_center'].value  # unit: MHz
        amplitude = res.params[f'p{i}_amplitude'].value
        # calculate value of bg at center
        slope = res.params['slope'].value
        intercept = res.params['intercept'].value
        constant = slope * center + intercept

        # get q
        freq_light = (freq_start * 1e3) + center  # unit: MHz
        q = freq_light / width

        # calculate contrast
        contrast = amplitude / (total_amp + constant)

        # save data
        all_center_curr.append(freq_light)
        all_q_curr.append(q)
        all_contrast_curr.append(contrast)

        if i != 0:
            text += "\n"
        text += rf"$\kappa_{i+1}$: {width:.3f} MHz"
        text += "\n"
        text += f"$Q_{i+1}$: {q:.3}"

    if device in center_by_device:
        center_by_device[device] += all_center_curr
        q_by_device[device] += all_q_curr
        contrast_by_device[device] += all_contrast_curr
    else:
        center_by_device[device] = all_center_curr
        q_by_device[device] = all_q_curr
        contrast_by_device[device] = all_contrast_curr

    if PLOT_ALL_RES:
        if PLOT_ALL_RES_TESTING:
            plt.plot(freq, transmission,
                     color=color, label='Data')
            plt.plot(freq[:-(SMOOTHING - 1)], smoothed_data,
                     color=color_smooth, label='Smoothed Data')
            plt.plot(freq, res.init_fit, 'r--',
                     label='Initial Fit')
            plt.plot(freq, res.best_fit, 'k--',
                     label='Fit')

            plt.gcf().text(0.95, 0.5, text,
                           ha='right', va='center', bbox=bbox)

            plt.title(f"Device {device} {wl:.3f} nm ({freq_start * 1e-3:.3f} THz) scan")
            plt.xlabel('Detuning (MHz)')
            plt.ylabel('Transmission (A.U.)')
            plt.legend()

            save_name = f"scan_{file_no}_wl_{wl:.3f}"
            save_name = save_name.replace(".", "_")
            save_name += ".png"
            output_path = os.path.join(OUTPUT_DIR, 'testing', f'{device}')
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            plt.savefig(os.path.join(output_path, save_name))
            plt.clf()

        plt.plot(freq, transmission,
                 color=color, label='Data')
        plt.plot(freq, res.best_fit, 'k--',
                 label='Fit')

        plt.gcf().text(0.95, 0.5, text,
                       ha='right', va='center', bbox=bbox)

        plt.title(f"Device {device} {wl:.3f} nm ({freq_start * 1e-3:.3f} THz) scan")
        plt.xlabel('Detuning (MHz)')
        plt.ylabel('Transmission (A.U.)')
        plt.legend()

        save_name = f"scan_{file_no}_wl_{wl:.3f}"
        save_name = save_name.replace(".", "_")
        save_name += ".png"
        output_path = os.path.join(OUTPUT_DIR, 'resonances', f'{device}')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        plt.savefig(os.path.join(output_path, save_name))
        plt.clf()


# save data for q
save_data = {"q": q_by_device,
             "centers": center_by_device,
             "contrasts": contrast_by_device}
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


# make a few relevant plots of all scan data
for device in center_by_device:
    print(f"Making final plots for device {device}.")

    centers = np.array(center_by_device[device])
    centers *= 1e-6  # convert to THz
    qs = np.array(q_by_device[device])
    contrasts = np.array(contrast_by_device[device])

    # make plot of q versus freq for each device
    # colorbar for contrast of each device
    for center, q, contrast in zip(centers, qs, contrasts):
        color_samp = colormap(contrast)
        plt.plot([center, center], [0, q], color=color_samp)
    plt.scatter(centers, qs, c=contrasts,
                cmap=colormap, vmin=0, vmax=1, zorder=3)
    plt.axhline(y=0, color='k')
    plt.title(f"Device {device}")
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Q Factor")
    plt.colorbar(label="Contrast")
    plt.tight_layout()

    if SAVE_DEVICE_PLOTS:
        plt.savefig(os.path.join(OUTPUT_DIR_ALL, f'device_{device:02d}.png'))
        plt.clf()
    else:
        plt.show()


    norm_power_1, norm_power_2 = calculate_enhancement(centers, qs, contrasts)
    n_photons_1, n_photons_2 = calculate_rates(centers, qs, contrasts, power=1)

    if PLOT_FIELD_ENHANCE:
        plt.stem(centers, np.sqrt(norm_power_1),
                 linefmt=color_factor, markerfmt=color_factor, basefmt='k')
        plt.axhline(y=0, color='k')
        plt.title(f"Device {device} Field Enhancement (Solution 1)")
        plt.xlabel("Frequency (THz)")
        plt.ylabel(r"$|F_0|^2$")
        plt.tight_layout()

        if SAVE_DEVICE_PLOTS:
            plt.savefig(os.path.join(OUTPUT_DIR_ALL, 'enhancement', f'device_{device:02d}_enhance_under.png'))
            plt.clf()
        else:
            plt.show()

        plt.stem(centers, np.sqrt(norm_power_2),
                 linefmt=color_factor, markerfmt=color_factor, basefmt='k')
        plt.axhline(y=0, color='k')
        plt.title(f"Device {device} Field Enhancement (Solution 2)")
        plt.xlabel("Frequency (THz)")
        plt.ylabel(r"$|F_0|^2$")
        plt.tight_layout()

        if SAVE_DEVICE_PLOTS:
            plt.savefig(os.path.join(OUTPUT_DIR_ALL, 'enhancement', f'device_{device:02d}_enhance_over.png'))
            plt.clf()
        else:
            plt.show()

    if PLOT_RATE:
        plt.stem(centers, n_photons_1,
                 linefmt=color_rate, markerfmt=color_rate, basefmt='k')
        plt.axhline(y=0, color='k')
        plt.title(f"Device {device} Estimated Pair Rate (Solution 1)")
        plt.xlabel("Frequency (THz)")
        plt.ylabel(r"Pairs $\mathrm{s}^{-1}$ $\mathrm{mW}^{-2}$")
        plt.yscale('log')
        plt.tight_layout()

        if SAVE_DEVICE_PLOTS:
            plt.savefig(os.path.join(OUTPUT_DIR_ALL, 'rates', f'device_{device:02d}_rate_under.png'))
            plt.clf()
        else:
            plt.show()

        plt.stem(centers, n_photons_2,
                 linefmt=color_rate, markerfmt=color_rate, basefmt='k')
        plt.axhline(y=0, color='k')
        plt.title(f"Device {device} Estimated Pair Rate (Solution 2)")
        plt.xlabel("Frequency (THz)")
        plt.ylabel(r"Pairs $\mathrm{s}^{-1}$ $\mathrm{mW}^{-2}$")
        plt.yscale('log')
        plt.tight_layout()

        if SAVE_DEVICE_PLOTS:
            plt.savefig(os.path.join(OUTPUT_DIR_ALL, 'rates', f'device_{device:02d}_rate_over.png'))
            plt.clf()
        else:
            plt.show()
