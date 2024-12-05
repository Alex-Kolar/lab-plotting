import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# collected data
BG = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
      "/Bulk_crystal/10mK/12032024/LASER_OFF.csv")
BEFORE_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
               "/Bulk_crystal/10mK/12032024/PREBURN.csv")
START_FREQ_BEFORE = 194809.212  # unit: GHz
END_FREQ_BEFORE = 194817.826  # unit: GHz
AFTER_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
              "/Bulk_crystal/10mK/12032024/AFTERBURN.csv")
START_FREQ_AFTER = 194810.216  # unit: GHz
END_FREQ_AFTER = 194818.918  # unit: GHz
PUMP_START_FREQ = 194813.912  # unit: GHz
PUMP_END_FREQ = 194814.912  # unit: GHz


# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
color_od = 'coral'

SHOW_PUMP = False


# REFERENCE DATA
df_laser_off = pd.read_csv(BG, header=10, skiprows=[11])
off_level = np.mean(df_laser_off['CH2'].astype(float).to_numpy())


# BEFORE PUMPING
df_before = pd.read_csv(BEFORE_DATA, header=10, skiprows=[11])

ramp = df_before['CH1'].astype(float).to_numpy()
transmission_before = df_before['CH2'].astype(float).to_numpy()
transmission_before -= off_level
print(off_level)

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
ramp = ramp[id_min:id_max]
transmission_before = transmission_before[id_min:id_max]

# convert to optical depth
bg = max(transmission_before)
od_before = np.log(bg / transmission_before)

# convert time to frequency
freq_before = np.linspace(0, (END_FREQ_BEFORE - START_FREQ_BEFORE), id_max - id_min)  # unit: GHz


# AFTER PUMPING
df_after = pd.read_csv(AFTER_DATA, header=10, skiprows=[11])

ramp = df_after['CH1'].astype(float).to_numpy()
transmission_after = df_after['CH2'].astype(float).to_numpy()
print(min(transmission_after))
transmission_after -= off_level
print(min(transmission_after))

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
ramp = ramp[id_min:id_max]
transmission_after = transmission_after[id_min:id_max]

# convert to optical depth
bg = max(transmission_after)
od_after = np.log(bg / transmission_after)

# convert time to frequency
freq_after = np.linspace(0, (END_FREQ_AFTER - START_FREQ_AFTER), id_max - id_min)  # unit: GHz


# plot optical depth and pumping
pump_start = PUMP_START_FREQ - START_FREQ_BEFORE
pump_end = PUMP_END_FREQ - START_FREQ_BEFORE
freq_diff = START_FREQ_AFTER - START_FREQ_BEFORE
plt.plot(freq_before, od_before, color='cornflowerblue', label="Before Pumping")
plt.plot(freq_after + freq_diff, od_after, color='coral', label="After Pumping")
plt.fill_between(freq_before, 0, 1, label="Pumping Region",
                 where=np.logical_and(freq_before >= pump_start, freq_before <= pump_end),
                 color='gray', alpha=0.2, transform=plt.gca().get_xaxis_transform())
plt.title("Hyperfine Polarization")
plt.xlabel(f"Frequency + {START_FREQ_BEFORE:.3f} (GHz)")
plt.ylabel("Optical Depth")
plt.legend(shadow=True)
plt.xlim((2, 6))
plt.ylim((0, 2))

plt.tight_layout()
plt.show()
