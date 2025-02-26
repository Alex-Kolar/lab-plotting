import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


DATA_BEFORE = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
               "/Bulk_crystal/10mK/02212025/POSTINIT1.csv")
DATA_COMB = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
             "/Bulk_crystal/10mK/02212025/5MINFULL.csv")
FREQ_BEFORE = (194811.364, 194814.077)
FREQ_COMB = (194811.575, 194814.207)
REF_FREQ = 194811  # unit: GHz
AOM_OFFSET = 0.680  # unit: GHz

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color_init = 'cornflowerblue'
color = 'coral'

SHOW_PUMP = True


# read initialized data
df_init = pd.read_csv(DATA_BEFORE, header=10, skiprows=[11])

ramp_init = df_init['CH1'].astype(float).to_numpy()
transmission_init = df_init['CH2'].astype(float).to_numpy()

id_min = np.argmin(ramp_init)
id_max = np.argmax(ramp_init)
ramp_init = ramp_init[id_min:id_max]
transmission_init = transmission_init[id_min:id_max]

# convert to optical depth
bg_init = max(transmission_init)
od_init = np.log(bg_init / transmission_init)

# convert time to frequency
freq_init = np.linspace(FREQ_BEFORE[0] - REF_FREQ, FREQ_BEFORE[1] - REF_FREQ, id_max-id_min)  # unit: GHz


# read comb data
df = pd.read_csv(DATA_COMB, header=10, skiprows=[11])

ramp = df['CH1'].astype(float).to_numpy()
transmission = df['CH2'].astype(float).to_numpy()

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
ramp = ramp[id_min:id_max]
transmission = transmission[id_min:id_max]

# convert to optical depth
bg = max(transmission)
od = np.log(bg / transmission)

# convert time to frequency
freq = np.linspace(FREQ_COMB[0] - REF_FREQ, FREQ_COMB[1] - REF_FREQ, id_max-id_min)  # unit: GHz


# plot transmission
plt.plot(freq_init, transmission_init, color=color_init,
         label='Initialized Spectrum')
plt.plot(freq, transmission, color=color,
         label='Comb Spectrum')
plt.xlabel(f"Frequency + {REF_FREQ + AOM_OFFSET:.3f} (GHz)")
plt.ylabel("Transmission (A.U.)")
plt.legend()
# plt.xlim((2.5, 6.5))

plt.tight_layout()
plt.show()


# plot optical depth
plt.plot(freq_init, od_init, color=color_init,
         label='Initialized Spectrum')
plt.plot(freq, od, color=color,
         label='Comb Spectrum')
plt.xlabel(f"Frequency + {REF_FREQ + AOM_OFFSET:.3f} (GHz)")
plt.ylabel("Optical Depth")
plt.legend()
# plt.xlim((2.5, 6.5))

plt.tight_layout()
plt.show()
