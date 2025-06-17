import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, ConstantModel


OFFRES = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
          "/New_mounted_device/10mK/2025_05_14/afc/afc_storage_experiment.npz")
DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/10mK/2025_05_14/afc")
FILE_FMT = "afc_storage_experiment_{}.npz"
# map file number to burn time
TIMES = {1: 5,
         2: 10,
         3: 20,
         4: 40}

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
xlim = (1.4, 1.6)
ylim = (0, 60)
PLOT_ALL_STORAGE = True

# fitting params
time_to_fit_offres = (0.9, 1.1)  # actual time in us
time_to_fit_storage = (0.4, 0.6)  # time in us relative to storage pulse


# gather data
offres_data = np.load(OFFRES, allow_pickle=True)
burn_times = []
burn_data = []
for file_no, time in TIMES.items():
    burn_times.append(time)
    file_path = os.path.join(DATA_DIR, FILE_FMT.format(file_no))
    data = np.load(file_path, allow_pickle=True)
    burn_data.append(data)

# fit offres data
idx_to_fit = np.where(np.logical_and(offres_data['bins'] > time_to_fit_offres[0],
                                     offres_data['bins'] < time_to_fit_offres[1]))[0]
model = GaussianModel() + ConstantModel()
offres_fit = model.fit(offres_data['counts'][idx_to_fit], x=offres_data['bins'][idx_to_fit],
                       amplitude=10000, sigma=0.005, center=1)
print(offres_fit.fit_report())

# plot fit
plt.plot(offres_data['bins'], offres_data['counts'],
         color='cornflowerblue',
         label='Input Storage Pulse')
plt.plot(offres_data['bins'][idx_to_fit], offres_fit.best_fit,
         ls='--', color='k',
         label='Gaussian Fit')

text = rf'Height: {offres_fit.params['height'].value:.0f} $\pm$ {offres_fit.params['height'].stderr:.0f}'
text += '\n'
text += rf'FWHM: {offres_fit.params['fwhm'].value * 1e3:.2f} $\pm$ {offres_fit.params['fwhm'].stderr * 1e3:.2f} ns'
ax = plt.gca()
plt.text(0.05, 0.95, text,
         ha='left', va='top',
         transform=ax.transAxes)

plt.title('Input Storage Pulse')
plt.xlabel(r'Time ($\mathrm{\mu}$s)')
plt.ylabel('Counts')
plt.legend(framealpha=1)
plt.xlim(time_to_fit_offres)

plt.tight_layout()
plt.show()

# fit other data
burn_fit = []
for time, data in zip(burn_times, burn_data):
    bins = data['bins'] - offres_fit.params['center'].value
    idx_to_fit = np.where(np.logical_and(bins > time_to_fit_storage[0],
                                         bins < time_to_fit_storage[1]))[0]
    counts = data['counts']
    fit = model.fit(counts[idx_to_fit], x=bins[idx_to_fit],
                    amplitude=1, sigma=0.005, center=0.5)

    # plot
    if PLOT_ALL_STORAGE:
        plt.plot(bins[idx_to_fit], counts[idx_to_fit],
                 color='coral',
                 label='Echo Pulse')
        plt.plot(bins[idx_to_fit], fit.best_fit,
                 ls='--', color='k',
                 label='Gaussian Fit')

        text = rf'Height: {fit.params['height'].value:.0f} $\pm$ {fit.params['height'].stderr:.0f}'
        text += '\n'
        text += rf'FWHM: {fit.params['fwhm'].value * 1e3:.2f} $\pm$ {fit.params['fwhm'].stderr * 1e3:.2f} ns'
        ax = plt.gca()
        plt.text(0.05, 0.95, text,
                 ha='left', va='top',
                 transform=ax.transAxes)

        plt.title(f'Echo Pulse with {time} Minute Burn')
        plt.xlabel(r'Time ($\mathrm{\mu}$s)')
        plt.ylabel('Counts')
        plt.legend(framealpha=1)

        plt.tight_layout()
        plt.show()


# fit input through values
burn_fit = []
for time, data in zip(burn_times, burn_data):
    bins = data['bins'] - offres_fit.params['center'].value

    # plot
    if PLOT_ALL_STORAGE:
        plt.plot(bins, data['counts'],
                 color='mediumpurple',
                 label='Input Pulse')

        plt.title(f'Input Pulse with {time} Minute Burn')
        plt.xlabel(r'Time ($\mathrm{\mu}$s)')
        plt.ylabel('Counts')
        plt.legend(framealpha=1)
        plt.xlim(-0.05, 0.05)

        plt.tight_layout()
        plt.show()


# # plot echos
# for time, data in zip(burn_times, burn_data):
#     plt.plot(data['bins'], data['counts'],
#              label=f'{time} Minute Burn')
#
# plt.xlabel(r'Time ($\mathrm{\mu}$s)')
# plt.ylabel('Counts')
# plt.legend()
# plt.xlim(xlim)
# plt.ylim(ylim)
#
# plt.tight_layout()
# plt.show()
