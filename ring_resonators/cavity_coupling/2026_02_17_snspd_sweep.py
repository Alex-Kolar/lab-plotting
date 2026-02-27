import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# data params2
DATA_DIR = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
            '/Mounted_device_mk_5/10mK/2026_02_17/initialization_snspd')
SWEEP_FILE = 'snspd_sweep_freqs.csv'
FILE_FMT = 'snspd_sweep_{:02}.txt'
FREQ_START = 194824.456
FREQ_END = 194833.227
AOM_FREQ = 0.6  # measured frequency of data from snspd_sweep_freqs is detuned by this amount from actual freq

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
OUTPUT_DIR = ('/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators'
              '/mounted_mk_5/10mK_coupling/2026_02_17_snspd')
XLIM = (2, 8)  # unit: GHz


# read cavity data files
reference_scan_df = pd.read_csv(os.path.join(DATA_DIR, 'snspd_sweep_reference.txt'), sep='\t')
time = reference_scan_df['time(ps)']
freq = np.linspace(0, FREQ_END-FREQ_START, len(time))
sweep_df = pd.read_csv(os.path.join(DATA_DIR, SWEEP_FILE))
scan_dfs = []
for row in sweep_df.iterrows():
    file_number = int(row[1]['File Number'])
    filename = FILE_FMT.format(file_number)
    df = pd.read_csv(os.path.join(DATA_DIR, filename), sep='\t')
    scan_dfs.append(df)

    sweep_start_freq = row[1]['Start Frequency (GHz)']
    sweep_end_freq = row[1]['End Frequency (GHz)']
    sweep_start_freq -= FREQ_START
    sweep_end_freq -= FREQ_START
    sweep_start_freq += AOM_FREQ
    sweep_end_freq += AOM_FREQ

    plt.plot(freq, df['counts'],
             color='cornflowerblue', label='Cavity Scan Data')
    plt.axvspan(sweep_start_freq, sweep_end_freq,
                alpha=0.2, color='gray', label='Initialization Sweep')
    plt.title(f'SNSPD Cavity Scan {file_number}')
    plt.xlabel(f'Frequency - {FREQ_START} GHz')
    plt.ylabel('Transmission (A.U.)')
    plt.legend(framealpha=1)
    plt.xlim(XLIM)
    plt.ylim(0, 9000)
    plt.savefig(os.path.join(OUTPUT_DIR, f'snspd_scan_{file_number}.png'))
    plt.clf()

# plotting all
# X, Y = np.meshgrid(freq, file_numbers)
# data = np.array([df['counts'].values for df in scan_dfs])
# plt.pcolormesh(X, Y, data, cmap='magma')
# plt.xlim(4, 6.5)
# plt.show()

fig, ax = plt.subplots()
ax.plot(freq, reference_scan_df['counts'] / max(reference_scan_df['counts']))
for i, df in enumerate(scan_dfs):
    norm_trans = df['counts']/max(df['counts'])
    ax.plot(freq, norm_trans + 0.5 * (i + 1))
ax.set_xlabel(f'Frequency - {FREQ_START} (GHz)')
ax.set_ylabel('Transmission')
ax.set_xlim(XLIM)

fig.tight_layout()
fig.show()
