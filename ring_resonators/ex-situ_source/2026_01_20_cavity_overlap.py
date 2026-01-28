import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


DATA = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
             '/Unmounted_device_roomtemp/2026_01_20/2CAV.csv')
LASER_OFF_PATH = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
                  '/Unmounted_device_roomtemp/2026_01_14/cavity_scan/background/data_000000.csv')
START_FREQ = 194827.613
END_FREQ = 194831.547
REF_FREQ = 194829.540

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
xlim = (-4, 4)
color_source = 'cornflowerblue'
color_memory = 'coral'


# read background data
bg_df = pd.read_csv(LASER_OFF_PATH)
transmission = bg_df['Data Voltage (V)'].astype(float)
bg_avg = np.min(transmission)
print(bg_avg)

df = pd.read_csv(DATA, header=10, skiprows=[11])

ramp = df['CH1'].astype(float).to_numpy()
transmission_source = df['CH2'].astype(float).to_numpy()
transmission_memory = df['CH3'].astype(float).to_numpy()

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
ramp = ramp[id_min:id_max]
transmission_source = transmission_source[id_min:id_max]
transmission_source -= bg_avg
transmission_memory = transmission_memory[id_min:id_max]
transmission_memory -= bg_avg

# convert time to frequency
freq = np.linspace(0, END_FREQ-START_FREQ, id_max-id_min)  # unit: GHz

# plotting
plt.plot(freq, transmission_source*10, color=color_source, label="Source Transmission")
plt.plot(freq, transmission_memory, color=color_memory, label="Memory Transmission")
plt.legend()
plt.xlabel(f"Detuning (GHz) from {START_FREQ} GHz")
plt.ylabel("Transmission (A.U.)")

plt.tight_layout()
plt.show()
