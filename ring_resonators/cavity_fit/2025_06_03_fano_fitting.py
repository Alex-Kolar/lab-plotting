import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import BreitWignerModel, ConstantModel


DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
        "/New_mounted_device/10mK/06032025/SDS00005.csv")
# pump resonance (file 7)
# TITLE = 'Pump Resonance'
# START_FREQ = 194986.727  # unit: GHz
# END_FREQ = 194995.339  # unit: GHz
# memory resonance (file 5)
TITLE = 'Signal Resonance'
START_FREQ = 194830.019
END_FREQ = 194838.606
# idler resonance (file 9)
# TITLE = 'Idler Resonance'
# START_FREQ = 195141.914
# END_FREQ = 195150.547

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
xlim = (-1, 1)
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
