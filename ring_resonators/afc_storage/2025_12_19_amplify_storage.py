import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# data params
DATA_DIR = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
            '/Mounted_device_mk_5/10mK/2025_12_19')
FILE_OFFRES = 'off_res_10min_pairs_aligned.txt'
FILE_PUMP_ON = 'on_res_10min_pairs_aligned.txt'
FILE_PUMP_ON_HIGHPOW = 'on_res_10min_pairs_aligned_highpow.txt'
FILE_NOPUMP = 'on_res_10min_nopairs.txt'
FILE_PUMP_OFF = 'on_res_10min_pairs_delayed.txt'
FILE_PUMP_OFF_HIGHPOW = 'on_res_10min_pairs_delayed_highpow.txt'

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
xlim = (1.0, 1.2)
ylim = (0, 100)
PLOT_HIGHPOWER = True


df_offres = pd.read_csv(os.path.join(DATA_DIR, FILE_OFFRES), sep='\t')
df_nopump = pd.read_csv(os.path.join(DATA_DIR, FILE_NOPUMP), sep='\t')
if PLOT_HIGHPOWER:
    df_pump = pd.read_csv(os.path.join(DATA_DIR, FILE_PUMP_ON_HIGHPOW), sep='\t')
    df_pump_delay = pd.read_csv(os.path.join(DATA_DIR, FILE_PUMP_OFF_HIGHPOW), sep='\t')
else:
    df_pump = pd.read_csv(os.path.join(DATA_DIR, FILE_PUMP_ON), sep='\t')
    df_pump_delay = pd.read_csv(os.path.join(DATA_DIR, FILE_PUMP_OFF), sep='\t')

time = df_offres['Time(ps)'] * 1e-6  # convert from ps to us
counts_offres = df_offres['Counts']
counts_pump = df_pump['Counts']
counts_delay = df_pump_delay['Counts']
counts_nopump = df_nopump['Counts']

plt.plot(time, counts_pump, label='Pump Aligned')
plt.plot(time, counts_delay, label='Pump Not Aligned')
plt.plot(time, counts_nopump, label='Pump Off')
plt.legend()
if PLOT_HIGHPOWER:
    plt.title('High Power Amplification Test')
else:
    plt.title('Amplification Test')
plt.xlabel(r'Time ($\mathrm{\mu}$s)')
plt.ylabel('Counts in 10 minutes')
plt.xlim(xlim)
plt.ylim(ylim)

plt.tight_layout()
plt.show()
