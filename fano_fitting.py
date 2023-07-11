import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import BreitWignerModel, ConstantModel


DATA = "/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators/Scan-05-19/1550_1704nm.csv"
WAVELENGTH = 1550.170  # unit: nm


df = pd.read_csv(DATA, header=5)
print(df)

time = df['Second'].astype(float)
ramp = df['Volt'].astype(float)
transmission = df['Volt.1'].astype(float)

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
time = time[id_min:id_max]
ramp = ramp[id_min:id_max]
transmission = transmission[id_min:id_max]

# convert time to frequency
FREQ_SCALE = 178.5  # unit: MHz/V
RAMP_AMP = 9.9  # unit: V (peak to peak)
freq = np.copy(time)
freq -= np.min(freq)
freq /= np.max(freq)
freq *= FREQ_SCALE * RAMP_AMP

# fitting
model = BreitWignerModel() + ConstantModel()
out = model.fit(transmission, x=freq,
                center=800)
print(out.fit_report())

# print out relevant information
sigma = out.params['sigma'].value * 1e6  # unit: Hz
print("Sigma:", sigma, "Hz")
freq_light = 3e8 / (WAVELENGTH * 1e-9)  # unit: Hz
print("Cavity freq:", freq_light, "Hz")
q = freq_light / sigma
print("Q:", q)


plt.plot(freq, transmission)
plt.plot(freq, out.best_fit, '--k')
plt.xlabel("Frequency (MHz)")
plt.ylabel("Transmission (AU)")

plt.tight_layout()
plt.show()

# for Milan measurements
time_length = np.max(time) - np.min(time)
print(time_length)
fit_width = 0.0014359
freq_width = (fit_width/time_length) * FREQ_SCALE * RAMP_AMP
print(freq_width)
print(freq_light * 1e-6 / freq_width)
