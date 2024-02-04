import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import BreitWignerModel, ConstantModel


# DATA = "/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators" \
#        "/Original_device/Coincidence Count Measurement/08022023/SDS00002.csv"
DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
        "/Planarized_device/cold_scan_12072023/D17_1535_914.csv")
WAVELENGTH = 1535.914  # unit: nm
SCAN_RANGE = 2  # unit: GHz

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'


df = pd.read_csv(DATA, header=10, skiprows=[11])

ramp = df['CH1'].astype(float)
transmission = df['CH2'].astype(float)

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
transmission = transmission[id_min:id_max]
freq = np.linspace(0, SCAN_RANGE * 1e3, id_max - id_min)  # unit: MHz

# fitting
model = BreitWignerModel(prefix='lf_') + BreitWignerModel(prefix='hf_') + ConstantModel()
out = model.fit(transmission, x=freq,
                lf_center=1100, lf_amplitude=0.02, lf_sigma=20, lf_q=0,
                hf_center=1250, hf_amplitude=0.02, hf_sigma=20, hf_q=0,
                slope=0, intercept=max(transmission))
print(out.fit_report())

# print out relevant information
freq_light = (3e8 / (WAVELENGTH * 1e-9)) * 1e-6  # unit: MHz
print("Low_Freq_Cavity freq:", freq_light, "MHz")

sigma = out.params['lf_sigma'].value  # unit: MHz
print("Low_Freq_Sigma:", sigma, "MHz")
q = freq_light / sigma
print("Low_Freq_Q:", q)

sigma = out.params['hf_sigma'].value  # unit: MHz
print("High_Freq_Sigma:", sigma, "MHz")
freq_light = (3e8 / (WAVELENGTH * 1e-9)) * 1e-6  # unit: MHz
q = freq_light / sigma
print("High_Freq_Q:", q)


plt.plot(freq, transmission, color=color, label="Transmission")
plt.plot(freq, out.best_fit, '--k', label="Fit")
# plt.plot(freq, out.init_fit, '--r', label="Initial Guess")
plt.title(f"Device 17 {WAVELENGTH} nm scan")
plt.xlabel("Detuning (MHz)")
plt.ylabel("Transmission (A.U.)")
plt.grid(True)
plt.legend(shadow=True)


plt.tight_layout()
plt.show()

# # for Milan measurements
# time_length = np.max(time) - np.min(time)
# print(time_length)
# fit_width = 0.0014359
# freq_width = (fit_width/time_length) * FREQ_SCALE * RAMP_AMP
# print(freq_width)
# print(freq_light * 1e-6 / freq_width)
