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
freq_before = np.linspace(0, (END_FREQ_BEFORE - START_FREQ_BEFORE), id_max - id_min)  # unit: GHz


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


# plot optical depth and pumping
pump_start = PUMP_START_FREQ - START_FREQ_BEFORE
pump_end = PUMP_END_FREQ - START_FREQ_BEFORE
freq_diff = START_FREQ_AFTER - START_FREQ_BEFORE
plt.plot(freq_before, od_before,
         color='cornflowerblue', label="Before Pumping")
plt.fill_between(freq_before, 0, 1,
                 label="Pumping Region",
                 where=np.logical_and(freq_before >= pump_start, freq_before <= pump_end),
                 color='gray', alpha=0.2, transform=plt.gca().get_xaxis_transform())
plt.plot(freq_after + freq_diff, od_after,
         color='coral', label="After Pumping")
plt.plot((freq_after+freq_diff)[idx_to_fit], res.best_fit,
         color='k', ls='--', label="Fit")

# add text
# horizontal_pos = 4.8
# vertical_pos = 2.5
# text = rf"FWHM: {res.params['fwhm'].value * 1e3:.2f} $\pm$ {res.params['fwhm'].stderr * 1e3:.2f} MHz"
# plt.text(horizontal_pos, vertical_pos, text,
#          ha='left', va='center', size=10)

plt.title("Hyperfine Polarization")
plt.xlabel(f"Frequency + {START_FREQ_BEFORE:.3f} (GHz)")
plt.ylabel("Optical Depth")
plt.legend(shadow=True, loc='upper left')
plt.xlim((2.5, 6))
plt.ylim((0, 3))

# for making labeled figure in CLEO abstract
# plt.tight_layout(rect=(0.05, 0, 1, 1))
plt.tight_layout()
if SAVE_FIG:
    plt.savefig(SAVENAME)
else:
    plt.show()


# plot just thermalized optical depth
pump_start = PUMP_START_FREQ - START_FREQ_BEFORE
pump_end = PUMP_END_FREQ - START_FREQ_BEFORE
freq_diff = START_FREQ_AFTER - START_FREQ_BEFORE
plt.plot(freq_before, od_before,
         color='cornflowerblue', label="Before Pumping")

plt.title("Thermalized Spectrum")
plt.xlabel(f"Frequency + {START_FREQ_BEFORE:.3f} (GHz)")
plt.ylabel("Optical Depth")
plt.xlim((2.5, 6))
plt.ylim((0, 2))

# for making labeled figure in CLEO abstract
# plt.tight_layout(rect=(0.05, 0, 1, 1))
plt.tight_layout()
if SAVE_FIG:
    plt.savefig(SAVENAME)
else:
    plt.show()

