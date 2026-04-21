import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import BreitWignerModel, ConstantModel


DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
        "/New_mounted_device/300K_no_erbium/01162025/SDS00013.csv")
SCAN_START = 195010.481  # unit: GHz
SCAN_END = 195014.446  # unit: GHz

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 14})
# xlim = (1000, 2000)
color = 'cornflowerblue'

# fitting params
fit_range = (0, 7000)  # entire range


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
# freq = np.linspace(0, SCAN_RANGE*1e3, id_max-id_min)  # unit: MHz
freq = np.linspace(0, SCAN_END-SCAN_START, id_max-id_min)  # unit: GHz
freq *= 1e3  # unit: MHz

# fitting
idx_to_fit = np.where(np.logical_and(freq >= fit_range[0],
                                     freq <= fit_range[1]))
idx_to_fit = idx_to_fit[0]
model = BreitWignerModel() + ConstantModel()
out = model.fit(transmission[idx_to_fit], x=freq[idx_to_fit],
                center=1500, amplitude=0.5, q=-0.2,
                c=0, sigma=100)
print(out.fit_report())

# print out relevant information
sigma = out.params['sigma'].value  # unit: MHz
sigma_err = out.params['sigma'].stderr
# print("Sigma:", sigma, "MHz")
print(f"Sigma: {sigma:0.3f} +/- {sigma_err:0.3f} MHz")
# freq_light = (3e8 / (WAVELENGTH * 1e-9)) * 1e-6  # unit: MHz
# freq_light = (SCAN_END + SCAN_START) / 2  # unit: GHz
# freq_light *= 1e3  # unit: MHz
freq_light = out.params['center'].value + (SCAN_START * 1e3)  # unit: MHz
print("Cavity freq:", freq_light, "MHz")
q = freq_light / sigma
print("Q:", q)


# plotting
fig, ax = plt.subplots(figsize=(8, 4))

freq_to_plot = freq[idx_to_fit] - out.params['center'].value
ax.plot(freq_to_plot, transmission[idx_to_fit], color=color, label="Transmission")
# ax.plot(freq_to_plot, out.best_fit, '--k', label="Fit")
ax.set_xlim(min(freq_to_plot), max(freq_to_plot))
# ax.grid(True)
# ax.legend(shadow=True)
wavelength = (3e8 / (freq_light * 1e6)) * 1e9
ax.set_title(f"Cavity Resonance at {wavelength:.3f} nm")
ax.set_xlabel("Detuning (MHz)")
ax.set_ylabel("Transmission (A.U.)")

# add relevant info
# label = r"$\Gamma$: {:0.3f} $\pm$ {:0.3f} MHz".format(
#     sigma, sigma_err)
# t = ax.text(0.05, 0.5, label,
#             horizontalalignment='left', verticalalignment='top')
# t.set_transform(ax.transAxes)

fig.tight_layout()
fig.show()

# # for Milan measurements
# time_length = np.max(time) - np.min(time)
# print(time_length)
# fit_width = 0.0014359
# freq_width = (fit_width/time_length) * FREQ_SCALE * RAMP_AMP
# print(freq_width)
# print(freq_light * 1e-6 / freq_width)
