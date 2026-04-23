import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel


OFFRES_DATA = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
               '/Mounted_device_mk_5/10mK/2026_04_08/afc/afc_storage_experiment_30db_offres.npz')
STORAGE_DATA = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
                '/Mounted_device_mk_5/10mK/2026_04_08/afc/afc_storage_experiment_30db_storage.npz')

edge_coupling_efficiency = 0.07
snspd_efficiency = 0.5

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color_offres = 'cornflowerblue'
color = 'coral'
color_2 = 'mediumpurple'
PLOT_PEAKS = False
xlim = (0.9, 2.1)


# read data
data_offres = np.load(OFFRES_DATA, allow_pickle=True)
data_echo = np.load(STORAGE_DATA, allow_pickle=True)

# plotting overlaid
fig, ax = plt.subplots(figsize=(5, 4), dpi=400)
time = data_offres['bins']
time_diff = time[1] - time[0]
idx_to_plot = np.where(np.logical_and(time > xlim[0], time < xlim[1]))[0]
plt.plot(time[idx_to_plot], data_offres['counts'][idx_to_plot],
         color=color_offres,
         label='Off-Resonant Pulse')
plt.plot(time[idx_to_plot], data_echo['counts'][idx_to_plot],
         color=color,
         label='Echo Pulse')

plt.title('Echo Measurement')
plt.xlabel(r'Time ($\mathrm{\mu}$s)')
plt.ylabel('Counts')
plt.legend(framealpha=1)
# plt.yscale('log')
plt.xlim(xlim)

plt.tight_layout()
plt.show()

# calculate efficiencies
idx_in = np.where(np.logical_and(time > 0.9, time < 1.1))[0]
idx_echo = np.where(np.logical_and(time > 1.9, time < 2.1))[0]
counts_in = np.sum(data_offres['counts'][idx_in])
counts_echo = np.sum(data_echo['counts'][idx_echo])
print('Counts In:', counts_in)
print('Counts Echo:', counts_echo)
print(f'Efficiency: {counts_echo/counts_in * 100:.2f}%')

# calculate pulse statistics
params = data_offres['storage parameters']
# print(params)
# pulse_freq = data_offres['storage parameters']['pulse_freq']
# integration_time = data_offres['storage parameters']['integration_time']
integration_time = 300
pulse_freq = 100e3
num_pulses = integration_time * pulse_freq
total_photons = counts_in / edge_coupling_efficiency / snspd_efficiency
photons_per_pulse = total_photons / num_pulses
print('Total photons on chip:', total_photons)
print('Photons per pulse:', photons_per_pulse)


# # analysis of echo shift
# model = GaussianModel()
# res_input = model.fit(data_offres['counts'][idx_in], x=time[idx_in],
#                       amplitude=300,
#                       center=0.975,
#                       sigma=0.01)
# res_echo = model.fit(data_echo['counts'][idx_echo], x=time[idx_echo],
#                      amplitude=30,
#                      center=1.975,
#                      sigma=0.01)
# fig, ax = plt.subplots()
# ax2 = ax.twinx()
# ax.plot(time[idx_in], data_offres['counts'][idx_in], color=color_offres)
# ax.plot(time[idx_in], res_input.best_fit,
#         color='k', ls='--')
# ax2.plot(time[idx_in[:-1]], data_echo['counts'][idx_echo], color=color)
# ax2.plot(time[idx_in[:-1]], res_echo.best_fit,
#          color='k', ls='--')
# ax.set_xlabel(r'Time ($\mathrm{\mu}$s)')
# ax.set_ylabel('Counts')
# ax2.set_ylabel('Counts')
# fig.tight_layout()
# fig.show()
#
# print(res_input.fit_report())
# print(res_echo.fit_report())
#
# time_diff = res_echo.best_values['center'] - res_input.best_values['center'] - 1
# print(f'Echo shift: {time_diff*1e3:.2f} ns')
