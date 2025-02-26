"""Data from December 9 2024

Sequence of data files:
1. REF: AOM scan taken with laser off-resonant at 1535 nm
2. PREBURN2: AOM scan taken on-resonant
3. AFTERBURN2: AOM scan taken on-resonant after burning with locked laser
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# data files
LASER_OFF = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
             "/Bulk_crystal/10mK/02072025/LASEROFF.csv")
BG_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
           "/Bulk_crystal/10mK/02072025/OFFRES.csv")
PREBURN = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
           "/Bulk_crystal/10mK/02072025/PREBURN.csv")
AFTERBURN = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
             "/Bulk_crystal/10mK/02072025/AFTERBURN.csv")
AFTERBURN_ZOOM = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
                  "/Bulk_crystal/10mK/02072025/AFTERBURNZOOM.csv")
AOM_RANGE = (67.855, 92.182)  # unit: MHz
CENTER_FREQ_ZOOM = 192812.69  # unit: GHz
CENTER_FREQ = 192812.71  # unit: GHz

# AFC params
N = 99
Delta = 1  # unit: MHz
Delta_f = 0.7  # unit: MHz
F = Delta / (Delta - Delta_f)

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica'})
color = 'cornflowerblue'
color_od = 'coral'

PLOT_ZOOM = False


# gather laser off level
df_laser_off = pd.read_csv(LASER_OFF, header=10, skiprows=[11])
off_level = np.mean(df_laser_off['CH2'].astype(float).to_numpy())

# gather data
df_bg = pd.read_csv(BG_DATA, header=10, skiprows=[11])
df_before = pd.read_csv(PREBURN, header=10, skiprows=[11])
if PLOT_ZOOM:
    df = pd.read_csv(AFTERBURN_ZOOM, header=10, skiprows=[11])
else:
    df = pd.read_csv(AFTERBURN, header=10, skiprows=[11])

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


# gather start level
ramp_before = df_before['CH1'].astype(float).to_numpy()
trans_before = df_before['CH2'].astype(float).to_numpy()
trans_before -= off_level

# plot ref level
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(ramp_before, color='yellow')
ax2.plot(trans_before, color='magenta')
ax.set_facecolor('black')
ax.set_title("Before Burning AOM scan")
ax.set_xlabel("Index")
ax.set_ylabel("Ramp (V)")
ax2.set_ylabel("Photodiode response (V)")
fig.tight_layout()
fig.show()


# convert after to transmission and OD
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
if PLOT_ZOOM:
    start = int(id_min_ref + (id_range / 4))
    trunc_range = int(id_range / 2) + 1
    trans_ref_trunc = trans_ref[start:(start + trunc_range)]
    trans_ref_trunc = np.repeat(trans_ref_trunc, 2)
    transmission = transmission / trans_ref_trunc[-1]
else:
    transmission = transmission / trans_ref[id_min_ref:(id_min_ref+id_range)]

# convert time to frequency
if PLOT_ZOOM:
    aom_total = (AOM_RANGE[1] - AOM_RANGE[0]) / 2
else:
    aom_total = AOM_RANGE[1] - AOM_RANGE[0]
freq = np.linspace(-aom_total/2, aom_total/2, id_max - id_min)  # unit: MHz

# convert to optical depth
od = np.log(1 / transmission)


# final plotting
plt.plot(freq, od,
         color=color)

plt.suptitle("Locked Laser AFC Hole Burning", fontsize=16)
plt.title(rf"$N$ = {N}, $\Delta$ = {Delta} MHz, $F$ = {F:0.3f}")
if PLOT_ZOOM:
    plt.xlabel(f"Detuning (MHz) from {CENTER_FREQ_ZOOM} (GHz)")
else:
    plt.xlabel(f"Detuning (MHz) from {CENTER_FREQ} (GHz)")
plt.ylabel("Optical Depth")

plt.xlim((-aom_total/2, aom_total/2))
# plt.xlim((-2, 2))

plt.tight_layout()
plt.show()
