import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import BreitWignerModel, ConstantModel
from scipy.signal import find_peaks


DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
        "/New_mounted_device/300K/07182024/SDS00019.csv")
SCAN_START = 195275.575  # unit: GHz
SCAN_END = 195284.118  # unit: GHz

# fitting params
SMOOTHING = 21
PEAK_THRESH = 0.85

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 10})
xlim = (-1000, 1000)
color = 'cornflowerblue'

# fitting params
fit_range = (5500, 8500)  # entire range


# moving average function for peaks
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


df = pd.read_csv(DATA, header=11)

# time = df['Second'].astype(float)
ramp = df['Volt'].astype(float)
transmission = df['Volt.1'].astype(float)

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
print(f"range: {id_min}:{id_max}")
ramp = ramp[id_min:id_max]
transmission = transmission[id_min:id_max]
ramp.reset_index(drop=True, inplace=True)
transmission.reset_index(drop=True, inplace=True)

# convert time to frequency
freq = np.linspace(0, SCAN_END-SCAN_START, id_max-id_min)  # unit: GHz
freq *= 1e3  # unit: MHz


# fitting
idx_to_fit = np.where(np.logical_and(freq >= fit_range[0],
                                     freq <= fit_range[1]))
idx_to_fit = idx_to_fit[0]
transmission = transmission[idx_to_fit]
transmission.reset_index(drop=True, inplace=True)
freq = freq[idx_to_fit]

# find peaks to determine number of fits
smoothed_data = moving_average(transmission.to_numpy(),
                               n=SMOOTHING)
peaks, peak_res = find_peaks(-smoothed_data,
                             prominence=0.015, distance=50, width=10)
peaks_to_keep = [p for p in peaks
                 if transmission[p] < (max(transmission)*PEAK_THRESH)]
print(f"\t\tnumber of peaks: {len(peaks_to_keep)}")
print(peaks_to_keep)
plt.plot(freq, transmission)
plt.plot(freq[peaks_to_keep], transmission[peaks_to_keep],
         ls='', marker='x')
plt.show()

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
out = model.fit(transmission, x=freq,
                **model_kwargs)

# print out relevant information
sigma = out.params['p0_sigma'].value  # unit: MHz
sigma_err = out.params['p0_sigma'].stderr
# print("Sigma:", sigma, "MHz")
print(f"Sigma: {sigma:0.3f} +/- {sigma_err:0.3f} MHz")
freq_light = out.params['p0_center'].value + (SCAN_START * 1e3)  # unit: MHz
print("Cavity freq:", freq_light, "MHz")
q = freq_light / sigma
print("Q:", q)


# plotting
fig, ax = plt.subplots(figsize=(3, 3), dpi=400)

freq_to_plot = freq - out.params['p0_center'].value
ax.plot(freq_to_plot, transmission, color=color, label="Transmission")
ax.plot(freq_to_plot, out.best_fit, '--k', label="Fit")

# fancy legend stuff
# ax.set_ylim(-20, -10)
# ax.legend(shadow=True)

wavelength = (3e8 / (freq_light * 1e6)) * 1e9
ax.set_title(f"Resonance at {wavelength:.3f} nm")
ax.set_xlabel("Detuning (MHz)")
ax.set_ylabel("Transmission (A.U.)")
ax.set_xlim(xlim)

fig.tight_layout()
fig.show()
