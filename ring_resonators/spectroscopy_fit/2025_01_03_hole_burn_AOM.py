"""Data from January 3 2025

Sequence of data files:
1. OFFRES: AOM scan taken with laser off-resonant at 1535 nm
2. PREBURN: AOM scan taken on-resonant
3. AFTERBURN: AOM scan taken on-resonant after burning with locked laser
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# data files
LASER_OFF = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
             "/Bulk_crystal/10mK/01032025/LASEROFF.csv")
BG_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
           "/Bulk_crystal/10mK/01032025/TEST/OFFRES.csv")
PREBURN = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
           "/Bulk_crystal/10mK/01032025/TEST/PREBURN.csv")
AFTERBURN = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
             "/Bulk_crystal/10mK/01032025/TEST/AFTERBURN.csv")
AOM_RANGE = (67.855, 92.182)  # unit: MHz
AOM_OFFSET = 0.680  # unit: GHz
CENTER_FREQ = 192812.7 + AOM_OFFSET  # unit: GHz

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color_before = 'cornflowerblue'
color_after = 'coral'


# gather laser off level
df_laser_off = pd.read_csv(LASER_OFF, header=10, skiprows=[11])
off_level = np.mean(df_laser_off['CH2'].astype(float).to_numpy())

# gather data
df_bg = pd.read_csv(BG_DATA, header=10, skiprows=[11])
df_before = pd.read_csv(PREBURN, header=10, skiprows=[11])
df_after = pd.read_csv(AFTERBURN, header=10, skiprows=[11])

# gather ref level
ramp_ref = df_bg['CH1'].astype(float).to_numpy()
trans_ref = df_bg['CH2'].astype(float).to_numpy()
trans_ref -= off_level

id_min_ref = np.argmin(ramp_ref)
id_max_ref = np.argmax(ramp_ref)

# plot ref level
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(ramp_ref, color='yellow')
ax2.plot(trans_ref, color='magenta')
ax.set_facecolor('black')
ax.set_title("Reference AOM scan")
ax.set_xlabel("Index")
ax.set_ylabel("Ramp (V)")
ax2.set_ylabel("Photodiode response (V)")
fig.tight_layout()
fig.show()


# convert before and after to transmission and OD
freqs = []
trans = []
ods = []
for df in [df_before, df_after]:
    ramp = df['CH1'].astype(float).to_numpy()
    transmission = df['CH2'].astype(float).to_numpy()
    transmission -= off_level

    # plot of data
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(ramp, color='yellow')
    ax2.plot(transmission, color='magenta')
    ax.set_facecolor('black')
    ax.set_title("AOM scan")
    ax.set_xlabel("Index")
    ax.set_ylabel("Ramp (V)")
    ax2.set_ylabel("Photodiode response (V)")
    fig.tight_layout()
    fig.show()

    id_min = np.argmin(ramp)
    id_max = np.argmax(ramp)
    transmission = transmission[id_min:id_max]

    # normalize
    id_range = id_max - id_min
    transmission = transmission / trans_ref[id_min_ref:(id_min_ref+id_range)]
    trans.append(transmission)

    # convert time to frequency
    aom_total = AOM_RANGE[1] - AOM_RANGE[0]
    freq = np.linspace(-aom_total/2, aom_total/2, id_max - id_min)  # unit: MHz
    freqs.append(freq)

    # convert to optical depth
    od = np.log(1 / transmission)
    ods.append(od)


# final plotting
plt.plot(freqs[0], ods[0],
         color=color_before, label='Before Burning')
plt.plot(freqs[1], ods[1],
         color=color_after, label='After Burning')

plt.xlim((-1, 1.5))
plt.ylim((0, 3.5))

plt.title("Locked Laser Hole Burning")
plt.xlabel(f"Detuning (MHz) from {CENTER_FREQ} GHz")
plt.ylabel("Optical Depth")
plt.legend(shadow=True)
plt.grid()

plt.tight_layout()
plt.show()
