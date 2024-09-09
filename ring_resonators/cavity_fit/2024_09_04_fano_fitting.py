import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import BreitWignerModel, ConstantModel


DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
        "/New_mounted_device/10mK/09042024/SDS00001.csv")
START_FREQ = 194811.382  # unit: GHz
END_FREQ = 194819.837  # unit: GHz

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
# xlim = (13000, 17000)
color = 'cornflowerblue'

# fitting params
# fit_range = (13000, 17000)
fit_range = (0, 8)


df = pd.read_csv(DATA, header=10, skiprows=[11])

ramp = df['CH1'].astype(float).to_numpy()
transmission = df['CH2'].astype(float).to_numpy()

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
ramp = ramp[id_min:id_max]
transmission = transmission[id_min:id_max]

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
sigma = out.params['sigma'].value  # unit: MHz
print("Sigma:", sigma, "GHz")
freq_light = START_FREQ + out.params['center'].value
print("Cavity freq:", freq_light, "GHz")
q = freq_light / sigma
print("Q:", q)


plt.plot(freq, transmission, color=color, label="Transmission")
plt.plot(freq, out.best_fit, '--k', label="Fit")
# plt.plot(freq, out.init_fit, '--r', label="Initial")
# plt.xlim(xlim)
plt.grid('on')
plt.legend(shadow=True)
plt.xlabel("Frequency (GHz)")
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
