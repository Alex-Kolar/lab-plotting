import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


SCAN_REF = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/Bulk_crystal/10mK/02122025/LASEROFF.csv")
SCAN_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
             "/Bulk_crystal/10mK/02122025/POSTINIT.csv")
MOD_REF = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
           "/Bulk_crystal/10mK/02122025/AmSpecHighPower/0.0.txt")
MOD_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/Bulk_crystal/10mK/02122025/AmSpecHighPower/194809180000000.0.txt")
FREQ_SCAN = (194809.898, 194817.505)
FREQ_MOD = 194809.18
AOM_OFFSET_SCAN = 0.600
AOM_OFFSET_MOD = 0.600

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color_od = 'coral'
color_mod = 'mediumpurple'


# read data
df_scan_ref = pd.read_csv(SCAN_REF, header=10, skiprows=[11])
df_scan_data = pd.read_csv(SCAN_DATA, header=10, skiprows=[11])
off_level = np.mean(df_scan_ref['CH2'].astype(float).to_numpy())
df_mod_ref = pd.read_csv(MOD_REF, sep=' ', names=['Frequency', 'Response'])
df_mod_data = pd.read_csv(MOD_DATA, sep=' ', names=['Frequency', 'Response'])

# process scan data
ramp = df_scan_data['CH1'].astype(float).to_numpy()
transmission = df_scan_data['CH2'].astype(float).to_numpy()
transmission -= off_level

# plot scan data
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(ramp, color='yellow')
ax2.plot(transmission, color='magenta')
ax1.set_facecolor('black')
ax1.set_title("Reference Piezo Scan")
ax1.set_xlabel("Index")
ax1.set_ylabel("Ramp (V)")
ax2.set_ylabel("Photodiode response (V)")
ax1.set_xlim(5000, 7000)
ax1.set_ylim(0.2, 0.6)
ax2.set_ylim(8, 10)
fig.tight_layout()
fig.show()

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
ramp = ramp[id_min:id_max]
transmission = transmission[id_min:id_max]

# convert to optical depth
bg = max(transmission)
od = np.log(bg / transmission)

# convert time to frequency
freq_scan = np.linspace(FREQ_SCAN[0] - FREQ_MOD, FREQ_SCAN[1] - FREQ_MOD,
                        id_max - id_min)  # unit: GHz
freq_scan += AOM_OFFSET_SCAN

# get AM spec data
normalized = df_mod_data['Response'] - df_mod_ref['Response']
freq = df_mod_ref['Frequency'] / 1e9  # convert to GHz
freq += AOM_OFFSET_MOD


# plotting
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(freq_scan, od, color=color_od)
ax2.plot(freq, normalized, color=color_mod)

ax1.set_title('Amplitude Modulation Spectroscopy February 12')
ax2.set_xlabel(f'Frequency - {FREQ_MOD} (GHz)')
ax1.set_ylabel('Optical Depth')
ax2.set_ylabel('Modulation Response (dB)')

plt.xlim(3, 6)

plt.tight_layout()
plt.show()
