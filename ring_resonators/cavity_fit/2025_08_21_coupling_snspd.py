import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import BreitWignerModel, LinearModel
from scipy.signal import find_peaks


DATA_OFF = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
            '/Mounted_device_mk_4/10mK/2025_08_21/cavityscan_5min_70db_200mT.txt')
DATA_ON = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
           '/Mounted_device_mk_4/10mK/2025_08_21/cavityscan_5min_70db_405mT.txt')
FREQ_START = 194810.775  # unit: GHz
FREQ_END = 194819.611


# load files
df_on = pd.read_csv(DATA_ON, sep='\t')
df_off = pd.read_csv(DATA_OFF, sep='\t')

time = df_off['time(ps)']
freq = np.linspace(0, FREQ_END-FREQ_START, len(time))


plt.plot(freq, df_off['counts'])
plt.plot(freq, df_on['counts'])
plt.xlim((4, 6))
plt.xlabel(f'Detuning (GHz) from {FREQ_START:.3f} GHz')
plt.ylabel('Counts')

plt.show()
