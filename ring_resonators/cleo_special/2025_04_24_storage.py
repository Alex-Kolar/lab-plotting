import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, ConstantModel


OFFRES_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
               "/New_mounted_device/10mK/2025_04_24/afc/testing/afc_storage_experiment_2.npz")
STORAGE_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
                "/New_mounted_device/10mK/2025_04_24/afc/testing/afc_storage_experiment.npz")
RESET_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
              "/New_mounted_device/10mK/2025_04_24/afc/testing/afc_storage_experiment_1.npz")
OFFCAV_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
               "/New_mounted_device/10mK/2025_05_06/afc/afc_storage_experiment_1.npz")


# fitting params
xrange_fit_store = (0.95, 1.08)
xrange_fit_echo = (1.08, 1.20)


# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 10})
color_offres = 'cornflowerblue'
color = 'coral'
color_reset = 'mediumpurple'
xlim = (0.95, 1.2)


# read data
data_offres = np.load(OFFRES_DATA)
data_echo = np.load(STORAGE_DATA)
data_reset = np.load(RESET_DATA)
data_offcav = np.load(OFFCAV_DATA)

# fitting data
time = data_echo['bins']
idx_to_fit = np.where(np.logical_and(time >= xlim[0], time <= xlim[1]))[0]

idx_to_fit_storage = np.where(np.logical_and(time >= xrange_fit_store[0], time <= xrange_fit_store[1]))
idx_to_fit_echo = np.where(np.logical_and(time >= xrange_fit_echo[0], time <= xrange_fit_echo[1]))

model = GaussianModel() + ConstantModel()
res_storage = model.fit(data_offres['counts'][idx_to_fit_storage], x=time[idx_to_fit_storage],
                        center=1.02, sigma=0.01, amplitude=1e4, c=1)
res_echo = model.fit(data_echo['counts'][idx_to_fit_echo], x=time[idx_to_fit_echo],
                     center=1.12, sigma=0.01, amplitude=20, c=1)
print(res_storage.fit_report())
print(res_echo.fit_report())

amplitude_store = res_storage.params['amplitude'].value
amplitude_echo = res_echo.params['amplitude'].value
print("efficiency:")
print(amplitude_echo/amplitude_store)


# plotting overlaid
time_diff = time[1] - time[0]
idx_to_plot = np.where(np.logical_and(time >= xlim[0], time <= xlim[1]))[0]
time_to_plot = time[idx_to_plot]
offres_to_plot = data_offres['counts'][idx_to_plot]
echo_to_plot = data_echo['counts'][idx_to_plot]
offcav_to_plot = data_offcav['counts'][idx_to_plot]

fig, ax = plt.subplots(figsize=(4, 3), dpi=400)

ax.bar(time_to_plot, offres_to_plot,
       width=time_diff,
       color=color_offres, alpha=0.5,
       label='Off-Resonant Pulse')
ax.bar(time_to_plot, echo_to_plot,
       width=time_diff,
       color=color, alpha=0.5,
       label='Echo Pulse')

ax.set_title('Echo Measurement')
ax.set_xlabel(r'Time ($\mathrm{\mu}$s)')
ax.set_ylabel('Counts')
ax.legend(framealpha=1)
ax.set_yscale('log')
ax.set_xlim(xlim)

fig.tight_layout()
fig.show()


# plotting overlaid of off-cavity
time_diff = time[1] - time[0]
idx_to_plot = np.where(np.logical_and(time >= xlim[0], time <= xlim[1]))[0]
time_to_plot = time[idx_to_plot]
offres_to_plot = data_offres['counts'][idx_to_plot]
echo_to_plot = data_echo['counts'][idx_to_plot]
offcav_to_plot = data_offcav['counts'][idx_to_plot]

fig, ax = plt.subplots(figsize=(5, 4), dpi=400)

ax.bar(time_to_plot, offcav_to_plot,
       width=time_diff,
       color=color_reset, alpha=0.5)

ax.set_title('Echo Measurement Off Cavity')
ax.set_xlabel(r'Time ($\mathrm{\mu}$s)')
ax.set_ylabel('Counts')
ax.set_yscale('log')
ax.set_xlim(xlim)

fig.tight_layout()
fig.show()


# # plotting overlaid with fit
# plt.bar(time_to_plot, offres_to_plot,
#         width=time_diff,
#         color=color_offres, alpha=0.5,
#         label='Off-Resonant Pulse')
# plt.bar(time_to_plot, echo_to_plot,
#         width=time_diff,
#         color=color, alpha=0.5,
#         label='Echo Pulse')
# plt.plot(time[idx_to_fit_storage], res_storage.best_fit,
#          color='k', ls='--')
# plt.plot(time[idx_to_fit_echo], res_echo.best_fit,
#          color='k', ls='--')
#
# plt.title('Echo Measurement')
# plt.xlabel(r'Time ($\mathrm{\mu}$s)')
# plt.ylabel('Counts')
# plt.legend(shadow=True)
# plt.yscale('log')
# plt.xlim(xlim)
#
# plt.tight_layout()
# plt.show()
