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

# plotting params
SAVE_FIG = False
SAVENAME = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
            "/bulk_crystal/10mK/pumping/12_03_polarized_pump_fit_OD.pdf")
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 10})
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


# plot just thermalized optical depth
fig, ax = plt.subplots(figsize=(5, 4), dpi=400)
plt.plot(freq_before, od_before,
         color='cornflowerblue', label="Before Pumping")

plt.title("Thermalized Spectrum (10 mK, 300 mT)")
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

