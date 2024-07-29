import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import BreitWignerModel, ConstantModel


DATA = "/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators" \
       "/Planarized_device/scan_4.csv"
WAVELENGTH = 1537.782  # unit: nm
SCAN_RANGE = 1.599  # unit: GHz

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'


df = pd.read_csv(DATA, header=1)
ramp = df['Volt'].astype(float)
transmission = df['Volt.1'].astype(float)

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
ramp = ramp[id_min:id_max]
transmission = transmission[id_min:id_max]

# convert time to frequency
freq = np.linspace(0, SCAN_RANGE*1e3, id_max-id_min)  # unit: MHz

# do fitting
model = BreitWignerModel(prefix='p1_') + BreitWignerModel(prefix='p2_') + ConstantModel()
out = model.fit(transmission, x=freq,
                p1_center=500, p2_center=1100, p1_q=0, p2_q=0)
print(out.fit_report())

# print out relevant information
# freq_light = (3e8 / (WAVELENGTH * 1e-9)) * 1e-6  # unit: MHz
freq_light = (195057.222 + 195055.623) / 2  # unit: GHz
freq_light *= 1e3  # unit: MHz
print("Cavity freq:", freq_light, "MHz")
sigma = out.params['p1_sigma'].value  # unit: MHz
print("Sigma1:", sigma, "MHz")
q = freq_light / sigma
print("Q:", q)
sigma = out.params['p2_sigma'].value  # unit: MHz
print("Sigma2:", sigma, "MHz")
q = freq_light / sigma
print("Q:", q)


plt.plot(freq, transmission, color=color, label="Transmission")
plt.plot(freq, out.best_fit, '--k', label="Fit")
# plt.xlim(xlim)
plt.grid('on')
plt.legend(shadow=True)
plt.xlabel("Frequency (MHz)")
plt.ylabel("Transmission (A.U.)")

plt.tight_layout()
plt.show()
