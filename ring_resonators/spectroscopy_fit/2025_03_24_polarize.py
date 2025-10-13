import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# collected data
DATA_START = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
              "/New_mounted_device/10mK/s2s_03242025/PREINIT1.csv")
DATA_LO = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
           "/New_mounted_device/10mK/s2s_03242025/POSTINIT1_LO.csv")
DATA_HI = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
           "/New_mounted_device/10mK/s2s_03242025/POSTINIT1_HI.csv")
FREQ_START = (194808.283, 194816.922)
FREQ_LO = (194807.812, 194816.532)
FREQ_HI = FREQ_LO
AOM_OFFSET = 0.600  # unit: GHz

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color_1 = 'cornflowerblue'
color_2 = 'coral'
color_3 = 'mediumpurple'
ref_freq = 194807  # unit: GHz


# BEFORE INITIALIZING
df_start = pd.read_csv(DATA_START, header=10, skiprows=[11])

ramp = df_start['CH1'].astype(float).to_numpy()
trans_start = df_start['CH2'].astype(float).to_numpy()

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
trans_start = trans_start[id_min:id_max]

# convert time to frequency
freq_start = np.linspace(FREQ_START[0]-ref_freq, FREQ_START[1]-ref_freq, id_max - id_min)  # unit: GHz
freq_start += AOM_OFFSET


# AFTER INITIALIZING (1)
df_lo = pd.read_csv(DATA_LO, header=10, skiprows=[11])

ramp = df_lo['CH1'].astype(float).to_numpy()
trans_lo = df_lo['CH2'].astype(float).to_numpy()

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
trans_lo = trans_lo[id_min:id_max]

# convert time to frequency
freq_lo = np.linspace(FREQ_LO[0]-ref_freq, FREQ_LO[1]-ref_freq, id_max - id_min)  # unit: GHz
freq_lo += AOM_OFFSET


# AFTER INITIALIZING (2)
df_hi = pd.read_csv(DATA_HI, header=10, skiprows=[11])

ramp = df_hi['CH1'].astype(float).to_numpy()
trans_hi = df_hi['CH2'].astype(float).to_numpy()

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
trans_hi = trans_hi[id_min:id_max]

# convert time to frequency
freq_hi = np.linspace(FREQ_HI[0]-ref_freq, FREQ_HI[1]-ref_freq, id_max - id_min)  # unit: GHz
freq_hi += AOM_OFFSET


# do plotting of transmission
plt.plot(freq_start, trans_start, color=color_1,
         label='Before Initialization')
plt.plot(freq_lo, trans_lo, color=color_2,
         label='After (Low Power)')
plt.plot(freq_hi, trans_hi, color=color_3,
         label='After (High Power)')

plt.title("Side-to-Side Initialization")
plt.xlabel(f"Frequency - {ref_freq} (GHz)")
plt.ylabel("Transmission (A.U.)")
plt.legend(shadow=True)
# plt.xlim(4, 8)

plt.tight_layout()
plt.show()
