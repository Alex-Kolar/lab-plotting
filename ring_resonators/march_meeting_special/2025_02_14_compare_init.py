import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import ConstantModel, VoigtModel


# collected data
BG = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
      "/Bulk_crystal/10mK/02142025/LASEROFF.csv")
DATA_START = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
              "/Bulk_crystal/10mK/01022025/PREINIT2.csv")
DATA_INIT_1 = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
               "/Bulk_crystal/10mK/02142025/POSTINIT.csv")
DATA_INIT_2 = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
               "/Bulk_crystal/10mK/02142025/POSTINIT2.csv")
FREQ_START = (194808.612, 194817.357)
FREQ_INIT_1 = (194810.024, 194818.577)
FREQ_INIT_2 = (194809.545, 194818.112)
AOM_OFFSET = 0.680  # unit: GHz

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 10})
color_before = 'cornflowerblue'
color_1 = 'coral'
color_2 = 'coral'
ref_freq = 194808

# fitting params
fit_start = 5.6
fit_end = 6.1


# REFERENCE DATA
df_laser_off = pd.read_csv(BG, header=10, skiprows=[11])
off_level = np.mean(df_laser_off['CH2'].astype(float).to_numpy())


# BEFORE PUMPING
df_before = pd.read_csv(DATA_START, header=10, skiprows=[11])

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
freq_before = np.linspace(FREQ_START[0]-ref_freq, FREQ_START[1]-ref_freq, id_max - id_min)  # unit: GHz
freq_before += AOM_OFFSET


# AFTER PUMPING (1)
df_after_1 = pd.read_csv(DATA_INIT_1, header=10, skiprows=[11])

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
freq_init_1 = np.linspace(FREQ_INIT_1[0]-ref_freq, FREQ_INIT_1[1]-ref_freq, id_max - id_min)  # unit: GHz
freq_init_1 += AOM_OFFSET


# AFTER PUMPING (2)
df_after_2 = pd.read_csv(DATA_INIT_2, header=10, skiprows=[11])

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
freq_init_2 = np.linspace(FREQ_INIT_2[0]-ref_freq, FREQ_INIT_2[1]-ref_freq, id_max - id_min)  # unit: GHz
freq_init_2 += AOM_OFFSET

# do fitting
freq_bounds = np.logical_and(freq_init_2 > fit_start, freq_init_2 < fit_end)
real_bounds = np.logical_and(freq_bounds, ~np.isnan(od_2))
idx_to_fit = np.where(real_bounds)[0]
model = VoigtModel() + ConstantModel()
res = model.fit(od_2[idx_to_fit], x=freq_init_2[idx_to_fit],
                center=5.75, sigma=0.25, c=0.2)
print(res.fit_report())
print(f"Center frequency: {ref_freq + res.params['center'].value}")


# do plotting
fig, ax = plt.subplots(figsize=(5, 4), dpi=400)

ax.plot(freq_before, od_before, color=color_before,
        label='Before Initialization')
ax.plot(freq_init_2, od_2, color=color_2,
        label='After Initialization')
ax.plot(freq_init_2[idx_to_fit], res.best_fit, ls='--', color='k',
        label='Fit')

ax.set_title("Hyperfine Polarization")
ax.set_xlabel(f"Frequency - {ref_freq} (GHz)")
ax.set_ylabel("Optical Depth")
ax.legend(shadow=True, loc='upper left')
ax.set_xlim(3.5, 7)
ax.set_ylim(0, 3)

fig.tight_layout()
plt.show()


# # add label for FWHM
# horizontal_pos = 5.95
# vertical_pos = 2.5
# text = rf"Height: OD {res.params['height'].value:.3f} $\pm$ {res.params['height'].stderr:.3f}"
# text += "\n"
# text += rf"FWHM: {res.params['fwhm'].value * 1e3:.3f} $\pm$ {res.params['fwhm'].stderr * 1e3:.3f} MHz"
# text += "\n"
# text += f"Center: {ref_freq + res.params['center'].value:.3f} $\pm$ {res.params['center'].stderr:.3f} GHz"
# plt.text(horizontal_pos, vertical_pos, text,
#          ha='left', va='center')
#
# ax.set_title("Hyperfine Polarization")
# ax.set_xlabel(f"Frequency - {ref_freq} (GHz)")
# ax.set_ylabel("Optical Depth")
# ax.legend(shadow=True, loc='upper left')
# ax.set_xlim(5.0, 6.6)
# ax.set_ylim(0, 3)
#
# fig.tight_layout()
# plt.show()
