import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


DATA_PATH = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
             '/Mounted_device_mk_5/10mK/cavity_coupling/11192025/CAVANDER.csv')
FREQ_START = 194825.400  # unit: GHz
FREQ_END = 194834.019  # unit: GHz

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color_cav = 'cornflowerblue'
color_er = 'coral'


data_df = pd.read_csv(DATA_PATH, header=10, skiprows=[11])

ramp = data_df['CH1'].astype(float)
erbium = data_df['CH2'].astype(float)
cavity = data_df['CH3'].astype(float)

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
erbium = erbium[id_min:id_max]
cavity.reset_index(drop=True, inplace=True)
cavity = cavity[id_min:id_max]
cavity.reset_index(drop=True, inplace=True)
freq = np.linspace(0, (FREQ_END-FREQ_START),
                   num=(id_max-id_min))  # unit: GHz


fig, axs = plt.subplots(2, 1)

axs[0].plot(freq, erbium,
            color=color_er)
axs[1].plot(freq, cavity*1e2,
            color=color_cav)

axs[0].set_title('Cavity-Ensemble Coupling')
axs[0].set_ylabel('Erbium Transmission (A.U.)')
axs[1].set_ylabel('Cavity Reflection (A.U.)')
axs[-1].set_xlabel(f'Detuning (GHz) from {FREQ_START} GHz')

fig.tight_layout()
plt.show()
