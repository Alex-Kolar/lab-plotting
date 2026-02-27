import os
import numpy as np
import pandas as pd
import glob
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt


# data
PL_DATA = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
           "/new_mounted/10mK_pl/all_fitted_decay/03252025/res_data.bin")
SPEC_DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
                 "/New_mounted_device/10mK/s2s_03252025")
FREQ_RANGE = (194807.649, 194816.359)
AOM_OFFSET = 0.600  # unit: GHz

# meta idx params
CUTOFF_IDX = 5

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color_spec = 'cornflowerblue'
color_pl = 'coral'
color_lifetime = 'mediumpurple'
ref_freq = 194807  # unit: GHz


# SPECTRUM DATA

# get spectrum data
files = glob.glob(os.path.join(SPEC_DATA_DIR, "PREINIT*.csv"))
dfs = [pd.read_csv(file, header=10, skiprows=[11]) for file in files]

# get ramp data from first file
df_start = dfs[0]
ramp = df_start['CH1'].astype(float).to_numpy()
id_min = np.argmin(ramp)
id_max = np.argmax(ramp)

# convert time to frequency
freq = np.linspace(FREQ_RANGE[0]-ref_freq, FREQ_RANGE[1]-ref_freq, id_max - id_min)  # unit: GHz
freq += AOM_OFFSET

# get transmission data for each scan
transmissions = []
for df in dfs:
    transmission = df['CH2']
    transmisison = transmission[id_min:id_max]
    transmissions.append(transmisison)

# get average
transmissions = np.array(transmissions)
avg_trans = np.sum(transmissions, axis=0) / len(transmissions)


# PL DATA

# get pl data
with open(PL_DATA, 'rb') as f:
    pl_data = pickle.load(f)
pl_freqs = np.array(pl_data['freqs'])
pl_freq_min = pl_data['freq_min']
pl_freqs += AOM_OFFSET + pl_freq_min - ref_freq  # put in terms of detuning, and add AOM offset
area, area_err = pl_data['area_fit']
tau, tau_err = pl_data['tau']


# PLOTTING

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 8))

axs[0].plot(freq, avg_trans,
            color=color_spec)
axs[1].errorbar(pl_freqs, area, yerr=area_err,
                marker='o', ls='', capsize=3, color=color_pl)
axs[2].errorbar(pl_freqs, tau*1e3, yerr=tau_err*1e3,
                marker='o', ls='', capsize=3, color=color_lifetime)

axs[0].set_title('Absorption Spectrum and PL')
axs[0].set_ylabel('Spectroscopy Transmission (A.U.)')
axs[1].set_ylabel('PL Area (A.U.)')
axs[2].set_ylabel('PL Lifetime (ms)')
axs[-1].set_xlabel(f'Frequency - {ref_freq} (GHz)')
axs[2].set_ylim((2.5, 10))

fig.tight_layout()
fig.show()
