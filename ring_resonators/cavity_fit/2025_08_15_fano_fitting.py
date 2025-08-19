import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import BreitWignerModel, ConstantModel


DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
        "/Mounted_device_mk_4/4K/2025_08_15/cavity_scan/device_14/data_000001.csv")
LASER_OFF_PATH = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
                  "/Mounted_device_mk_3/4K/2025_07_22/background/data_000000.csv")
TITLE = 'Test Resonance'
START_FREQ = 194809.319
END_FREQ = 194817.821

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
xlim = (-1, 1)
color = 'cornflowerblue'

# fitting params
# fit_range = (13000, 17000)
fit_range = (0, 8)


# read background data
bg_df = pd.read_csv(LASER_OFF_PATH)
transmission = bg_df['Data Voltage (V)'].astype(float)
bg_avg = np.min(transmission)

df = pd.read_csv(DATA)

ramp = df['Ramp Voltage (V)'].astype(float).to_numpy()
transmission = df['Data Voltage (V)'].astype(float).to_numpy()

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
ramp = ramp[id_min:id_max]
transmission = transmission[id_min:id_max]
transmission -= bg_avg

# convert time to frequency
freq = np.linspace(0, (END_FREQ - START_FREQ), id_max-id_min)  # unit: GHz

# fitting
# idx_to_fit = np.where(np.logical_and(freq >= fit_range[0],
#                                      freq <= fit_range[1]))
# idx_to_fit = idx_to_fit[0]
model = ConstantModel() + BreitWignerModel()
out = model.fit(transmission, x=freq,
                center=4.475, amplitude=0.2, q=0,
                c=0.1, sigma=0.4)
print(out.fit_report())

# print out relevant information
kappa = out.params['sigma'].value  # unit: MHz
print("kappa:", kappa, "GHz")
freq_light = START_FREQ + out.params['center'].value
print("Cavity freq:", freq_light, "GHz")
q = freq_light / kappa
print("Q:", q)
amplitude = out.params['amplitude'].value
constant = out.params['c'].value
contrast = amplitude / (amplitude + constant)
print("Contrast:", contrast)
print("Reflection:", 1-contrast)


freq = freq - out.params['center'].value
plt.plot(freq, transmission, color=color, label="Transmission")
plt.plot(freq, out.best_fit, '--k', label="Fit")
# plt.plot(freq, out.init_fit, '--r', label="Initial")
plt.xlim(xlim)
plt.legend(shadow=True)
plt.title(TITLE)
plt.xlabel("Detuning (GHz)")
plt.ylabel("Transmission (A.U.)")

plt.tight_layout()
plt.show()
