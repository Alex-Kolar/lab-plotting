import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


OFFRES_DATA = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
               '/Mounted_device_mk_5/10mK/2026_02_24/afc/afc_multimode_offres.npz')
STORAGE_DATA_CONSTRUCTIVE = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
                             '/Mounted_device_mk_5/10mK/2026_02_24/afc/afc_multimode_storage_locked_constructive.npz')
STORAGE_DATA_DESTRUCTIVE = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
                            '/Mounted_device_mk_5/10mK/2026_02_24/afc/afc_multimode_storage_locked_destructive.npz')
interference_window = (1.97-0.05, 1.97+0.05)

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color_offres = 'cornflowerblue'
color = 'coral'
PLOT_PEAKS = False
xlim = (1.97-0.2, 1.97+0.2)


# read data
data_offres = np.load(OFFRES_DATA, allow_pickle=True)
data_echo_constructive = np.load(STORAGE_DATA_CONSTRUCTIVE, allow_pickle=True)
data_echo_destructive = np.load(STORAGE_DATA_DESTRUCTIVE, allow_pickle=True)
time = data_offres['bins']

# calculate visibility
int_idx = np.where(np.logical_and(time > interference_window[0], time < interference_window[1]))[0]
counts_constructive = np.sum(data_echo_constructive['counts'][int_idx])
counts_destructive = np.sum(data_echo_destructive['counts'][int_idx])
visibility = (counts_constructive - counts_destructive) / (counts_constructive + counts_destructive)
print(f'Visibility: {visibility*100:.2f}%')

# plotting interference
time_diff = time[1] - time[0]
plt.bar(time, data_echo_constructive['counts'],
        width=time_diff,
        color='cornflowerblue', alpha=0.5,
        label=r'$\Delta\phi = 0$')
plt.bar(time, data_echo_destructive['counts'],
        width=time_diff,
        color='coral', alpha=0.5,
        label=r'$\Delta\phi = \pi$')
plt.axvline(x=interference_window[0], color='k', linestyle='--')
plt.axvline(x=interference_window[1], color='k', linestyle='--')

plt.title('Echo Measurement')
plt.xlabel(r'Time ($\mathrm{\mu}$s)')
plt.ylabel('Counts')
plt.legend(framealpha=1)
# plt.yscale('log')
plt.xlim(xlim)

plt.tight_layout()
plt.show()
