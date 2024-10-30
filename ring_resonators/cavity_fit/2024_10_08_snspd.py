import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import LorentzianModel, ConstantModel


INIT_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
             "/New_mounted_device/10mK/snspd_10082024/no_pump_Histogram_2024-10-08_17-16-50.txt")
PUMP_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
             "/New_mounted_device/10mK/snspd_10082024/pump_Histogram_2024-10-08_17-51-26.txt")

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Arial',
                     'font.size': 12})
xlim = (4e9, 6e9)


df_init = pd.read_csv(INIT_DATA, sep='\t')
df_pump = pd.read_csv(PUMP_DATA, sep='\t')


# plotting
plt.plot(df_init['time(ps)'], df_init['counts'],
         color='cornflowerblue')
plt.title("Before Pumping")
plt.xlabel("Tagger Time (ps)")
plt.ylabel("Counts")
plt.xlim(xlim)

plt.tight_layout()
plt.show()

plt.plot(df_pump['time(ps)'], df_pump['counts'],
         color='cornflowerblue')
plt.title("After Pumping")
plt.xlabel("Tagger Time (ps)")
plt.ylabel("Counts")
plt.xlim(xlim)

plt.tight_layout()
plt.show()

plt.plot(df_init['time(ps)'], df_pump['counts']/df_init['counts'],
         color='coral')
plt.title("Ratio After/Before Pumping")
plt.xlabel("Tagger Time (ps)")
plt.xlim(xlim)

plt.tight_layout()
plt.show()
