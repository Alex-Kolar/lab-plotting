import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# collected data
BG = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
      "/New_mounted_device/10mK/06182025/SDS00002.csv")
DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
        "/New_mounted_device/10mK/06182025/SDS00001.csv")
FREQ = (194831.476, 194835.424)
AOM_OFFSET = 0.680  # unit: GHz


# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color_1 = 'cornflowerblue'
ref_freq = 194833.531 + AOM_OFFSET


# REFERENCE DATA
df_bg = pd.read_csv(BG, header=10, skiprows=[11])

ramp = df_bg['CH1'].astype(float).to_numpy()
transmission = df_bg['CH2'].astype(float).to_numpy()

# convert time to frequency
id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
transmission_bg = transmission[id_min:id_max]
freq_bg = np.linspace(FREQ[0]-ref_freq, FREQ[1]-ref_freq, id_max - id_min)  # unit: GHz
freq_bg += AOM_OFFSET


# AFTER BURNING (1)
df_burn = pd.read_csv(DATA, header=10, skiprows=[11])

ramp = df_burn['CH1'].astype(float).to_numpy()
transmission = df_burn['CH2'].astype(float).to_numpy()

# convert time to frequency
id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
transmission = transmission[id_min:id_max]
freq = np.linspace(FREQ[0]-ref_freq, FREQ[1]-ref_freq, id_max - id_min)  # unit: GHz
freq += AOM_OFFSET


# plotting
plt.plot(freq_bg, transmission_bg)
plt.plot(freq, transmission)

plt.xlim(-0.5, 0.5)

plt.show()


norm = transmission[:-5] / transmission_bg
plt.plot(freq_bg, norm)

plt.xlim(-0.5, 0.5)

plt.show()
