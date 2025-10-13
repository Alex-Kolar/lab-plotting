import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


MOD_REF = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
           "/New_mounted_device/4K/05162025/AmSpec/0.0.txt")
MOD_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/4K/05162025/AmSpec/194805000000000.0.txt")
FREQ_MOD = 194804.967

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color_mod = 'cornflowerblue'


# read data
df_mod_ref = pd.read_csv(MOD_REF, sep=' ', names=['Frequency', 'Response'])
df_mod_data = pd.read_csv(MOD_DATA, sep=' ', names=['Frequency', 'Response'])

# get AM spec data
normalized = df_mod_data['Response'] - df_mod_ref['Response']
freq = df_mod_ref['Frequency'] / 1e9  # convert to GHz


# plotting
plt.plot(freq, normalized,
         color=color_mod)

plt.xlabel(f'Frequency - {FREQ_MOD} (GHz)')
plt.ylabel('Modulation Response (dB)')
plt.xlim((4, 8))

plt.tight_layout()
plt.show()
