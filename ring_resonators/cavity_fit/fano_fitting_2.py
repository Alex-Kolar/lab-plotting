import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import BreitWignerModel, ConstantModel


# DATA = "/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators" \
#        "/Original_device/Coincidence Count Measurement/08022023/SDS00002.csv"
DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
        "/Planarized_device/cold_scan_12012023/cold_1535p914.csv")
WAVELENGTH = 1535.48  # unit: nm
SCAN_RANGE = 30.623  # unit: GHz

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
xlim = (13000, 17000)
color = 'cornflowerblue'

# fitting params
fit_range = (13000, 17000)


df = pd.read_csv(DATA, header=1)
print(df)

# time = df['Second'].astype(float)
ramp = df['Volt'].astype(float)
transmission = df['Volt.1'].astype(float)

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
# time = time[id_min:id_max]
ramp = ramp[id_min:id_max]
transmission = transmission[id_min:id_max]

# convert time to frequency
freq = np.linspace(0, SCAN_RANGE*1e3, id_max-id_min)  # unit: MHz

# fitting
idx_to_fit = np.where(np.logical_and(freq >= fit_range[0],
                                     freq <= fit_range[1]))
idx_to_fit = idx_to_fit[0]
model = BreitWignerModel() + ConstantModel()
out = model.fit(transmission[idx_to_fit], x=freq[idx_to_fit],
                center=15300, amplitude=0.006, q=0.2,
                c=0.012, sigma=50)
print(out.fit_report())

# print out relevant information
sigma = out.params['sigma'].value  # unit: MHz
print("Sigma:", sigma, "MHz")
freq_light = (3e8 / (WAVELENGTH * 1e-9)) * 1e-6  # unit: MHz
print("Cavity freq:", freq_light, "MHz")
q = freq_light / sigma
print("Q:", q)


plt.plot(freq, transmission, color=color, label="Transmission")
plt.plot(freq[idx_to_fit], out.best_fit, '--k', label="Fit")
plt.plot(freq[idx_to_fit], out.init_fit, '--r', label="Initial")
plt.xlim(xlim)
plt.grid('on')
plt.legend(shadow=True)
plt.xlabel("Frequency (MHz)")
plt.ylabel("Transmission (A.U.)")

plt.tight_layout()
plt.show()

# # for Milan measurements
# time_length = np.max(time) - np.min(time)
# print(time_length)
# fit_width = 0.0014359
# freq_width = (fit_width/time_length) * FREQ_SCALE * RAMP_AMP
# print(freq_width)
# print(freq_light * 1e-6 / freq_width)
