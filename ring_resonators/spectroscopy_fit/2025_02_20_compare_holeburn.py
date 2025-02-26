import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import ConstantModel, VoigtModel


# collected data
BG = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
      "/Bulk_crystal/10mK/02202025/BACKGROUND.csv")
DATA_1 = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
          "/Bulk_crystal/10mK/02202025/HOLE2.csv")
DATA_2 = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
          "/Bulk_crystal/10mK/02202025/HOLE3.csv")
DATA_3 = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
          "/Bulk_crystal/10mK/02202025/HOLE4.csv")
FREQ_1 = (194811.428, 194814.102)
FREQ_2 = (194811.451, 194814.111)
FREQ_3 = (194811.502, 194814.162)
AOM_OFFSET = 0.680  # unit: GHz

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color_1 = 'cornflowerblue'
color_2 = 'coral'
color_3 = 'mediumpurple'
ref_freq = 194811


# REFERENCE DATA
df_laser_off = pd.read_csv(BG, header=10, skiprows=[11])
off_level = np.mean(df_laser_off['CH2'].astype(float).to_numpy())


# AFTER BURNING (1)
df_after_1 = pd.read_csv(DATA_1, header=10, skiprows=[11])

ramp = df_after_1['CH1'].astype(float).to_numpy()
transmission_1 = df_after_1['CH2'].astype(float).to_numpy()
print(min(transmission_1))
transmission_1 -= off_level
print(min(transmission_1))

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
ramp = ramp[id_min:id_max]
transmission_1 = transmission_1[id_min:id_max]

# convert to optical depth
bg = max(transmission_1)
od_1 = np.log(bg / transmission_1)

# convert time to frequency
freq_init_1 = np.linspace(FREQ_1[0]-ref_freq, FREQ_1[1]-ref_freq, id_max - id_min)  # unit: GHz
freq_init_1 += AOM_OFFSET


# AFTER PUMPING (2)
df_after_2 = pd.read_csv(DATA_2, header=10, skiprows=[11])

ramp = df_after_2['CH1'].astype(float).to_numpy()
transmission_2 = df_after_2['CH2'].astype(float).to_numpy()
print(min(transmission_2))
transmission_2 -= off_level
print(min(transmission_2))

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
ramp = ramp[id_min:id_max]
transmission_2 = transmission_2[id_min:id_max]

# convert to optical depth
bg = max(transmission_2)
od_2 = np.log(bg / transmission_2)

# convert time to frequency
freq_init_2 = np.linspace(FREQ_2[0]-ref_freq, FREQ_2[1]-ref_freq, id_max - id_min)  # unit: GHz
freq_init_2 += AOM_OFFSET


# AFTER PUMPING (3)
df_after_3 = pd.read_csv(DATA_3, header=10, skiprows=[11])

ramp = df_after_3['CH1'].astype(float).to_numpy()
transmission_3 = df_after_3['CH2'].astype(float).to_numpy()
print(min(transmission_3))
transmission_3 -= off_level
print(min(transmission_3))

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
ramp = ramp[id_min:id_max]
transmission_3 = transmission_3[id_min:id_max]

# convert to optical depth
bg = max(transmission_3)
od_3 = np.log(bg / transmission_3)

# convert time to frequency
freq_init_3 = np.linspace(FREQ_3[0]-ref_freq, FREQ_3[1]-ref_freq, id_max - id_min)  # unit: GHz
freq_init_3 += AOM_OFFSET


# do plotting
# plt.plot(freq_init_1, transmission_1, color=color_1,
#          label='Hole Burn 1')
# plt.plot(freq_init_2, transmission_2, color=color_2,
#          label='Hole Burn 2')
plt.plot(freq_init_3, transmission_3, color=color_3,
         label='Hole Burn 3')

plt.title("Hole Burning")
plt.xlabel(f"Frequency - {ref_freq} (GHz)")
plt.ylabel("Transmission (A.U.)")
plt.legend(shadow=True, loc='upper left')
plt.grid(True)
plt.xlim(2.4, 2.6)
# plt.ylim(0, 4)

plt.tight_layout()
plt.show()


# do plotting
# plt.plot(freq_init_1, od_1, color=color_1,
#          label='Hole Burn 1')
# plt.plot(freq_init_2, od_2, color=color_2,
#          label='Hole Burn 2')
plt.plot(freq_init_3, od_3, color=color_3,
         label='Hole Burn 3')

plt.title("Hole Burning")
plt.xlabel(f"Frequency - {ref_freq} (GHz)")
plt.ylabel("Optical Depth")
plt.legend(shadow=True, loc='upper left')
plt.grid(True)
plt.xlim(2.4, 2.6)
plt.ylim(0, 4)

plt.tight_layout()
plt.show()
