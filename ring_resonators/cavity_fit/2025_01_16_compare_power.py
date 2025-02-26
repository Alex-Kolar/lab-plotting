import glob
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import BreitWignerModel, ConstantModel
from scipy.signal import find_peaks
import pickle


LOW_POWER_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
                  "/New_mounted_device/300K_no_erbium/01162025/LOWPOWER.csv")
HIGH_POWER_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
                   "/New_mounted_device/300K_no_erbium/01162025/HIGHPOWER.csv")
LOW_POWER_FREQ = (195064.451, 195066.323)  # unit: GHz
HIGH_POWER_FREQ = (195064.649, 195066.518)  # unit: GHz

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color_low = 'cornflowerblue'
color_high = 'coral'


min_freq = min(LOW_POWER_FREQ[0], HIGH_POWER_FREQ[0])

# find and read oscilloscope files
df_low = pd.read_csv(LOW_POWER_DATA, header=10, skiprows=[11])
df_high = pd.read_csv(HIGH_POWER_DATA, header=10, skiprows=[11])

# get data from files
ramp_low = df_low['CH1'].astype(float)
transmission_low = df_low['CH2'].astype(float)
transmission_low = transmission_low / max(transmission_low)  # normalize
id_min = np.argmin(ramp_low)
id_max = np.argmax(ramp_low)
transmission_low = transmission_low[id_min:id_max]
transmission_low.reset_index(drop=True, inplace=True)
freq_low = np.linspace(LOW_POWER_FREQ[0]-min_freq, LOW_POWER_FREQ[1]-min_freq,
                       num=(id_max-id_min))  # unit: MHz

ramp_high = df_high['CH1'].astype(float)
transmission_high = df_high['CH2'].astype(float)
transmission_high = transmission_high / max(transmission_high)  # normalize
id_min = np.argmin(ramp_high)
id_max = np.argmax(ramp_high)
transmission_high = transmission_high[id_min:id_max]
transmission_high.reset_index(drop=True, inplace=True)
freq_high = np.linspace(HIGH_POWER_FREQ[0]-min_freq, HIGH_POWER_FREQ[1]-min_freq,
                        num=(id_max-id_min))  # unit: MHz


# plotting
plt.plot(freq_low, transmission_low, label='Scan',
         color=color_low)
plt.plot(freq_high, transmission_high, label='Scan +20 dB',
         color=color_high)

plt.title('Cavity Power Modulation')
plt.xlabel(f'Detuning (GHz) from {min_freq:.3f} GHz')
plt.ylabel('Normalized Transmission')
plt.legend(shadow=True)
plt.xlim((0.5, 1.5))
plt.grid(True)

plt.tight_layout()
plt.show()

