import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, ConstantModel


FILENAME = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/10mK/coincidence_2025_05_27/Correlation_2025-05-28_09-56-17.txt")
FILENAME_PULSE = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
                  "/New_mounted_device/10mK/coincidence_2025_05_27/SDS00001.csv")


# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
xlim_range = 100  # size of x lims
color_pulse = 'cornflowerblue'
color = 'coral'

# fitting params for pulse
fit_range = (-100, 150)
fwhm_const = 2 * np.sqrt(2 * np.log(2))  # ratio of FWHM to sigma


# NOTE: first line of csv needs extra tab added
# otherwise the columns will not be read properly
df = pd.read_csv(FILENAME, sep='\t')
coincidence = df["Counts"]
time = df["Time(ps)"]  # unit: ps
time *= 1e-3  # unit: ns
time_diff = time[1] - time[0]  # spacing of histogram

df_pulse = pd.read_csv(FILENAME_PULSE, header=10, skiprows=[11])
time_pulse = df_pulse['Source'].astype(float).to_numpy()
pulse = df_pulse['CH2'].astype(float).to_numpy()
time_pulse *= 1e9  # convert to ns

# fit pulse
idx_to_fit = np.where(np.logical_and(time_pulse > fit_range[0], time_pulse < fit_range[1]))[0]
time_to_fit = time_pulse[idx_to_fit]
pulse_to_fit = pulse[idx_to_fit]
model = GaussianModel() + ConstantModel()
res = model.fit(pulse_to_fit, x=time_to_fit,
                center=20, sigma=20, c=-0.1, amplitude=100)
# get data
sigma = res.params['sigma'].value
fwhm = fwhm_const * sigma
sigma_convolve = np.sqrt(2) * sigma
fwhm_convolve = fwhm_const * sigma_convolve
print(f"Fitted FWHM: {fwhm} ns")
print(f"Convolved FWHM: {fwhm_convolve} ns")


# plot pulse
# plotting
plt.plot(time_pulse, pulse, color=color_pulse,
         label='Data')
plt.plot(time_to_fit, res.best_fit,
         color='k', ls='--',
         label='Fit')

plt.title('Pump Pulse Shape')
plt.xlabel('Time (ns)')
plt.ylabel('Pulse Amplitude (A.U.)')
plt.legend(shadow=True)
plt.xlim(-100, 200)

plt.tight_layout()
plt.show()


# plot coincidence
fig, ax = plt.subplots()

ax.bar(time, coincidence, width=time_diff, color=color)

center = time[np.argmax(coincidence)]
xlim = (center - xlim_range/2, center + xlim_range/2)
ax.set_xlim(xlim)
ax.set_xlabel("Timing Offset (ns)")
ax.set_ylabel("Coincidence Counts")
ax.set_title("Two-Photon Coincidence")

fig.tight_layout()
fig.show()
