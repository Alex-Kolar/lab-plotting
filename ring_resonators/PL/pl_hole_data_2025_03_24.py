import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import ExponentialModel, ConstantModel


BEFORE_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
               "/New_mounted_device/10mK/s2s_pl_03242025/PL_waveguide_unlock_0db.txt")
AFTER_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
              "/New_mounted_device/10mK/s2s_pl_03242025/PL_waveguide_unlock_0db_afterholeburn.txt")
CUTOFF_IDX = 10

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
color2 = 'coral'
color3 = 'mediumpurple'


df_before = pd.read_csv(BEFORE_DATA, sep='\t')
time_before = df_before['time(ps)'][CUTOFF_IDX:]
time_before /= 1e9
counts_before = df_before['counts'][CUTOFF_IDX:]

# do fitting of before pl
model_before = ExponentialModel() + ConstantModel()
res_before = model_before.fit(counts_before, x=time_before)
print(res_before.fit_report())

df_after = pd.read_csv(AFTER_DATA, sep='\t')
time_after = df_after['time(ps)'][CUTOFF_IDX:]
time_after /= 1e9
counts_after = df_after['counts'][CUTOFF_IDX:]

# do fitting of before pl
model_after = ExponentialModel() + ConstantModel()
res_after = model_after.fit(counts_after, x=time_after)
print(res_after.fit_report())


plt.plot(time_before, counts_before,
         color=color, alpha=0.5,
         label='Before Holeburning')
plt.plot(time_before, res_before.best_fit,
         color=color, ls='--',
         label='Before Holeburning Fit')
plt.plot(time_after, counts_after,
         color=color2, alpha=0.5,
         label='After Holeburning')
plt.plot(time_after, res_after.best_fit,
         color=color2, ls='--',
         label='After Holeburning Fit')

plt.xlabel('Time (ms)')
plt.ylabel('Counts')
plt.legend(shadow=True)

plt.tight_layout()
plt.show()
