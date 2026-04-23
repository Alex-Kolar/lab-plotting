import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


DATA_DIR = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
            '/Mounted_device_mk_5/10mK/2026_02_24/afc/afc_dual_comb_no_phase_offset')
one_pulse_offres = 'one_pulse_offres.txt'
one_pulse_storage = 'one_pulse_3_00pi.txt'
two_pulse_offres = 'two_pulse_offres.txt'
pi_to_plot_0 = 0.0
pi_to_plot_1 = 1.0

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
xlim = (1.99-0.2, 1.99+0.2)
color_0 = 'cornflowerblue'
color_1 = 'coral'


# gather data
storage_files = glob.glob('two_pulse_*pi.txt', root_dir=DATA_DIR)
storage_files.sort()

# read data
df_0 = None
df_1 = None
for file in storage_files:
    file_path = os.path.join(DATA_DIR, file)
    file_parts = file.split('_')
    pi_fraction = float(file_parts[2]) + (float(file_parts[3][:2])/100)
    if pi_fraction == pi_to_plot_0:
        df_0 = pd.read_csv(file_path, sep='\t')
    elif pi_fraction == pi_to_plot_1:
        df_1 = pd.read_csv(file_path, sep='\t')

time = df_0['time(ps)']
time /= 1e6  # convert to us
counts_0 = df_0['counts']
counts_1 = df_1['counts']

idx_to_plot = (time > xlim[0]) & (time < xlim[1])
time = time[idx_to_plot]
counts_0 = counts_0[idx_to_plot]
counts_1 = counts_1[idx_to_plot]

fig, ax = plt.subplots(figsize=(5, 3), dpi=400)
plt.plot(time, counts_0,
         color=color_0,
         label=r'$\Delta\phi = 0$')
plt.plot(time, counts_1,
         color=color_1,
         label=r'$\Delta\phi = \pi$')
# plt.ylim(0, 800)
plt.xlim(xlim)
plt.title('Two-Comb Interference')
plt.xlabel(r'Time ($\mu$s)')
plt.ylabel(r'Counts')
plt.legend()
plt.tight_layout()
plt.show()
