import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import VoigtModel, ConstantModel


# collected data
BG = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
      "/Bulk_crystal/10mK/01022025/LASEROFF.csv")
BEFORE_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
               "/Bulk_crystal/10mK/01022025/PREINIT2.csv")
AOM_OFFSET = 0.680
START_FREQ_BEFORE = 194808.612 + AOM_OFFSET  # unit: GHz (+ AOM offset)
END_FREQ_BEFORE = 194817.357 + AOM_OFFSET  # unit: GHz
AFTER_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
              "/Bulk_crystal/10mK/01022025/POSTINIT.csv")
START_FREQ_AFTER = 194809.568 + AOM_OFFSET  # unit: GHz
END_FREQ_AFTER = 194818.099 + AOM_OFFSET  # unit: GHz
PUMP_START_FREQ = 194813.281 + AOM_OFFSET  # unit: GHz
PUMP_END_FREQ = 194814.184 + AOM_OFFSET  # unit: GHz

# fitting region
fit_start = 3.3
fit_end = 3.9

# plotting params
SAVE_FIG = False
SAVENAME = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
            "/bulk_crystal/10mK/pumping/12_03_polarized_pump_fit_OD.pdf")
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 14})
color = 'cornflowerblue'
color_od = 'coral'
ref_freq = 194808


# REFERENCE DATA
df_laser_off = pd.read_csv(BG, header=10, skiprows=[11])
off_level = np.mean(df_laser_off['CH2'].astype(float).to_numpy())


# BEFORE PUMPING
df_before = pd.read_csv(BEFORE_DATA, header=10, skiprows=[11])

ramp = df_before['CH1'].astype(float).to_numpy()
transmission_before = df_before['CH2'].astype(float).to_numpy()
transmission_before -= off_level

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
ramp = ramp[id_min:id_max]
transmission_before = transmission_before[id_min:id_max]

# convert to optical depth
bg = max(transmission_before)
od_before = np.log(bg / transmission_before)

# convert time to frequency
freq_before = np.linspace(START_FREQ_BEFORE-ref_freq,
                          END_FREQ_BEFORE-ref_freq, id_max - id_min)  # unit: GHz


# AFTER PUMPING
df_after = pd.read_csv(AFTER_DATA, header=10, skiprows=[11])

ramp = df_after['CH1'].astype(float).to_numpy()
transmission_after = df_after['CH2'].astype(float).to_numpy()
transmission_after -= off_level

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
ramp = ramp[id_min:id_max]
transmission_after = transmission_after[id_min:id_max]

# convert to optical depth
bg = max(transmission_after)
od_after = np.log(bg / transmission_after)

# convert time to frequency
freq_after = np.linspace(0, (END_FREQ_AFTER - START_FREQ_AFTER), id_max - id_min)  # unit: GHz

# do fitting
freq_bounds = np.logical_and(freq_after > fit_start, freq_after < fit_end)
real_bounds = np.logical_and(freq_bounds, ~np.isnan(od_after))
idx_to_fit = np.where(real_bounds)[0]
model = VoigtModel() + ConstantModel()
res = model.fit(od_after[idx_to_fit], x=freq_after[idx_to_fit],
                center=3.55, sigma=0.25, c=0.2)
print(res.fit_report())
print(f"Center frequency: {START_FREQ_AFTER + res.params['center'].value}")


# plot just thermalized optical depth
fig, ax = plt.subplots(figsize=(8, 6))
plt.plot(freq_before, od_before,
         color='cornflowerblue', label="Before Pumping")

plt.title("Thermalized Spectrum")
plt.xlabel(f"Frequency + {ref_freq} (GHz)")
plt.ylabel("Optical Depth")
plt.xlim((3.5, 7))
plt.ylim((0, 2))

# for making labeled figure in CLEO abstract
# plt.tight_layout(rect=(0.05, 0, 1, 1))
plt.tight_layout()
if SAVE_FIG:
    plt.savefig(SAVENAME)
else:
    plt.show()

