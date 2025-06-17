import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel


OFFRES = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
          "/New_mounted_device/10mK/2025_05_20/afc/afc_storage_experiment_1.npz")
CAVITY_ONLY = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
               "/New_mounted_device/10mK/2025_05_20/afc/afc_storage_experiment.npz")
CAVITY_IONS = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
               "/New_mounted_device/10mK/2025_05_20/afc/afc_storage_experiment_3.npz")
CAVITY_AFC = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
              "/New_mounted_device/10mK/2025_05_20/afc/afc_storage_experiment_4.npz")
kappa = 628  # units: MHz

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})


# gather data
offres_data = np.load(OFFRES, allow_pickle=True)
cavity_data = np.load(CAVITY_ONLY, allow_pickle=True)
cavity_ion_data = np.load(CAVITY_IONS, allow_pickle=True)
cavity_afc_data = np.load(CAVITY_AFC, allow_pickle=True)

# fit data
model = GaussianModel()
res_offres = model.fit(offres_data['counts'], x=offres_data['bins'],
                       sigma=0.01, center=0.99, amplitude=700)
res_cavity = model.fit(cavity_data['counts'], x=cavity_data['bins'],
                       sigma=0.01, center=0.99, amplitude=300)
res_cavion = model.fit(cavity_ion_data['counts'], x=cavity_ion_data['bins'],
                       sigma=0.01, center=0.99, amplitude=300)
res_cavafc = model.fit(cavity_afc_data['counts'], x=cavity_afc_data['bins'],
                       sigma=0.01, center=0.99, amplitude=300)

# gather area data
range = (0.96, 1.02)
idx = np.where(np.logical_and(offres_data['bins'] > range[0], offres_data['bins'] < range[1]))[0]
offres_counts = np.sum(offres_data['counts'][idx])
cavity_counts = np.sum(cavity_data['counts'][idx])
cavity_ion_counts = np.sum(cavity_ion_data['counts'][idx])
cavity_afc_counts = np.sum(cavity_afc_data['counts'][idx])

# display fit data
print('Off Resonant Pulse')
print(f'\tHeight: {res_offres.params['height'].value:.0f}')
print(f'\tFWHM: {res_offres.params['fwhm'].value:.3f}')
print(f'\tNumerical area: {offres_counts:.3f}')
print('Cavity Pulse')
print(f'\tHeight: {res_cavity.params['height'].value:.0f}')
print(f'\tFWHM: {res_cavity.params['fwhm'].value:.3f}')
print(f'\tNumerical area: {cavity_counts:.3f}')
print('Cavity + Ions Pulse')
print(f'\tHeight: {res_cavion.params['height'].value:.0f}')
print(f'\tFWHM: {res_cavion.params['fwhm'].value:.3f}')
print(f'\tNumerical area: {cavity_ion_counts:.3f}')
print('Cavity + AFC Pulse')
print(f'\tHeight: {res_cavafc.params['height'].value:.0f}')
print(f'\tFWHM: {res_cavafc.params['fwhm'].value:.3f}')
print(f'\tNumerical area: {cavity_afc_counts:.3f}')


# # calculate values
# R = cavity_ion_counts / offres_counts
# kappa_in = (kappa/2) * (1 + np.sqrt(R))
# R_ion = cavity_afc_counts / offres_counts
# kappa_ion = (2*kappa_in) / (1 + np.sqrt(R_ion)) - kappa
# print('\n')
# print(f'kappa_in: {kappa_in}')
# print(f'kappa_ion: {kappa_ion}')

R = cavity_ion_counts / offres_counts
kappa_in = (kappa/2) * (1 - np.sqrt(R))
R_afc = cavity_afc_counts / offres_counts
kappa_afc = kappa - (2*kappa_in) / (1 - np.sqrt(R_afc))
print(f'kappa: {kappa:.3f}')
print(f'kappa_in: {kappa_in:.3f}')
print(f'kappa_afc: {kappa_afc:.3f}')


# plotting of all reflected pulses together
plt.plot(offres_data['bins'], offres_data['counts'],
         label='Off Resonant')
plt.plot(cavity_data['bins'], cavity_data['counts'],
         label='Cavity')
plt.plot(cavity_ion_data['bins'], cavity_ion_data['counts'],
         label='Cavity and Ions')
plt.plot(cavity_afc_data['bins'], cavity_afc_data['counts'],
         label='Cavity and AFC')

plt.xlim((0.95, 1.05))
plt.tight_layout()
plt.legend()
plt.show()


# plotting of each with its respective fit
plt.plot(offres_data['bins'] - res_offres.params['center'].value,
         offres_data['counts'],
         label='Data')
plt.plot(offres_data['bins'] - res_offres.params['center'].value,
         res_offres.best_fit,
         ls='--', color='k', label='Fit')
plt.title('Off Resonant Pulse')
plt.xlim((-0.05, 0.05))
plt.tight_layout()
plt.show()

plt.plot(cavity_data['bins'] - res_cavity.params['center'].value,
         cavity_data['counts'],
         label='Data')
plt.plot(cavity_data['bins'] - res_cavity.params['center'].value,
         res_cavity.best_fit,
         ls='--', color='k', label='Fit')
plt.title('Cavity Pulse')
plt.xlim((-0.05, 0.05))
plt.tight_layout()
plt.show()

plt.plot(cavity_ion_data['bins'] - res_cavion.params['center'].value,
         cavity_ion_data['counts'],
         label='Data')
plt.plot(cavity_ion_data['bins'] - res_cavion.params['center'].value,
         res_cavion.best_fit,
         ls='--', color='k', label='Fit')
plt.title('Cavity + Ions Pulse')
plt.xlim((-0.05, 0.05))
plt.tight_layout()
plt.show()

plt.plot(cavity_afc_data['bins'] - res_cavafc.params['center'].value,
         cavity_afc_data['counts'],
         label='Data')
plt.plot(cavity_afc_data['bins'] - res_cavafc.params['center'].value,
         res_cavafc.best_fit,
         ls='--', color='k', label='Fit')
plt.title('Cavity + AFC Pulse')
plt.xlim((-0.05, 0.05))
plt.tight_layout()
plt.show()
