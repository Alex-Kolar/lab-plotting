import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


DATA_UNLOCK = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
               '/interferometer/PM100LOG/unlock_stp.CSV')
DATA_LOCK = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
             '/interferometer/PM100LOG/lock_stp.CSV')
DATA_VACUUM = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
               '/interferometer/PM100LOG/lock_vacuum.CSV')
DATA_VACUUM_2 = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
                '/interferometer/PM100LOG/lock_vacuum_2.CSV')

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
xlim = (-1, 1)
color = 'cornflowerblue'


# import data
df_unlock = pd.read_csv(DATA_UNLOCK,
                        names=['Power (W)', 'Time (ms)'], skiprows=[0, 1, 2, 3], sep='\t')
df_lock = pd.read_csv(DATA_LOCK,
                      names=['Power (W)', 'Time (ms)'], skiprows=[0, 1, 2, 3], sep='\t')
df_vacuum = pd.read_csv(DATA_VACUUM,
                        names=['Power (W)', 'Time (ms)'], skiprows=[0, 1, 2, 3], sep='\t')
df_vacuum_2 = pd.read_csv(DATA_VACUUM_2,
                          names=['Power (W)', 'Time (ms)'], skiprows=[0, 1, 2, 3], sep='\t')


plt.plot(df_unlock['Time (ms)']/1e3, df_unlock['Power (W)']*1e3,
         label='Unlocked')
plt.plot(df_lock['Time (ms)']/1e3, df_lock['Power (W)']*1e3,
         label='Locked')
plt.plot(df_vacuum['Time (ms)']/1e3, df_vacuum['Power (W)']*1e3,
         label='Locked and Vacuum')
plt.plot(df_vacuum_2['Time (ms)']/1e3, df_vacuum_2['Power (W)']*1e3,
         label='Locked and Vacuum Overnight')

plt.xlabel('Time (s)')
plt.ylabel('Power (mW)')
plt.legend()
plt.tight_layout()
plt.show()
