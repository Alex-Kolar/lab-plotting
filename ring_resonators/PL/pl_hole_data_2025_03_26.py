import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import ExponentialModel, ConstantModel


HOLE_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
             "/New_mounted_device/10mK/PL_holeburn_2025_03_26/test_locked6.npy")

# data params
integration_time = 300  # unit: s
pl_freq = 20  # unit: Hz
pl_period = (1/pl_freq)*1e3  # unit: ms

# processing params
idx_to_skip = 5
smooth = False
smoothing = 20  # unit: * pl_period
smoothing_fit_cutoff = 20  # unit: s


# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
color2 = 'coral'
color3 = 'mediumpurple'


# get data
data = np.load(HOLE_DATA)
pl_data = np.sum(data, axis=0)
pl_time = np.linspace(0, pl_period, num=len(pl_data))
counts_data = np.sum(data, axis=1)
counts_time = np.linspace(0, integration_time, num=len(counts_data))
if smooth:
    counts_smoothed = np.convolve(counts_data, np.ones(smoothing)/smoothing, mode='valid')
    counts_time_smoothed = counts_time[:-(smoothing-1)]
else:
    counts_smoothed = counts_data
    counts_time_smoothed = counts_time



# fit data
model_pl = ExponentialModel() + ConstantModel()
res_pl = model_pl.fit(pl_data[idx_to_skip:], x=pl_time[idx_to_skip:],
                      decay=5, amplitude=80)

model_counts = ExponentialModel() + ConstantModel()
idx_to_fit = np.where(counts_time_smoothed < smoothing_fit_cutoff)[0]
res_counts = model_counts.fit(counts_smoothed[idx_to_fit],
                              x=counts_time_smoothed[idx_to_fit],
                              decay=1, amplitude=6, c=4)


# plot PL
plt.plot(pl_time[idx_to_skip:], pl_data[idx_to_skip:],
         color=color)
plt.plot(pl_time[idx_to_skip:], res_pl.best_fit,
         color='k', ls='--')
# add text for decay
t1 = res_pl.params['decay'].value
t1_err = res_pl.params['decay'].stderr
text = rf'$T_1$ = {t1:.3f} $\pm$ {t1_err:.3f} ms'
ax = plt.gca()
plt.text(0.95, 0.95, text,
         ha='right', va='top',
         transform=ax.transAxes)
plt.title('Photoluminescence Measurement')
plt.xlabel('Time (ms)')
plt.ylabel('Counts')
plt.xlim(0, pl_period)

plt.tight_layout()
plt.show()


# plot hole burning
plt.plot(counts_time_smoothed, counts_smoothed,
         color=color2)
plt.title('Hole Burn Measurement')
plt.xlabel('Time (s)')
plt.ylabel('Counts per Excitation Pulse')
plt.xlim(0, smoothing_fit_cutoff)

plt.tight_layout()
plt.show()


# plot hole burning with fit
plt.plot(counts_time_smoothed, counts_smoothed,
         color=color2)
plt.plot(counts_time_smoothed[idx_to_fit], res_counts.best_fit,
         color='k', ls='--')
# add text for decay
t1 = res_counts.params['decay'].value
t1_err = res_counts.params['decay'].stderr
text = rf'$\tau$ = {t1:.3f} $\pm$ {t1_err:.3f} s'
ax = plt.gca()
plt.text(0.95, 0.95, text,
         ha='right', va='top',
         transform=ax.transAxes)
plt.title('Hole Burn Measurement')
plt.xlabel('Time (s)')
plt.ylabel('Counts per Excitation Pulse')
plt.xlim(0, smoothing_fit_cutoff)

plt.tight_layout()
plt.show()


# plot of all photons
X, Y = np.meshgrid(pl_time, counts_time)
plt.pcolormesh(X, Y, data, cmap='gray')
plt.xlabel('PL Time (ms)')
plt.ylabel('Total Experiment Time (s)')

plt.tight_layout()
plt.show()
