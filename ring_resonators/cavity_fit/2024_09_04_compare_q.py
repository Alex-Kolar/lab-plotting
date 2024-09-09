import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import BreitWignerModel, ConstantModel


DATA_ON = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
           "/New_mounted_device/10mK/09032024/SDS00002.csv")
FREQ_ON = (194811.486, 194819.973)  # unit: GHz
DATA_OFF = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/10mK/09042024/SDS00001.csv")
FREQ_OFF = (194811.382, 194819.837)  # unit: GHz
FREQ_ERR = 0.01  # unit: GHz

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'


df_on = pd.read_csv(DATA_ON, header=10, skiprows=[11])
df_off = pd.read_csv(DATA_OFF, header=10, skiprows=[11])


# do fitting of on-resonant
ramp = df_on['CH1'].astype(float).to_numpy()
transmission = df_on['CH2'].astype(float).to_numpy()

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
ramp = ramp[id_min:id_max]
transmission = transmission[id_min:id_max]

# convert time to frequency
freq = np.linspace(0, (FREQ_ON[1] - FREQ_ON[0]), id_max-id_min)  # unit: GHz

model = ConstantModel() + BreitWignerModel()
out = model.fit(transmission, x=freq,
                center=4.475, amplitude=0.2, q=0,
                c=0.1, sigma=0.4)

# print out relevant information
print("On-resonant data:")
sigma = out.params['sigma'].value  # unit: GHz
sigma_err = out.params['sigma'].stderr
freq_light = FREQ_ON[0] + out.params['center'].value  # unit: GHz
freq_err = np.sqrt(FREQ_ERR**2 + out.params['center'].stderr**2)
q_on = freq_light / sigma
q_on_err = q_on * np.sqrt((freq_err/freq_light)**2 + (sigma_err/sigma)**2)
print(f"\tSigma: {sigma:.6f} +/- {sigma_err:.6f} GHz")
print(f"\tCavity freq: {freq_light:.6f} +/- {freq_err:.6f} GHz")
print(f"\tQ: {q_on:.0f} +/- {q_on_err:0.0f}")

# plotting
plt.plot(freq, transmission, color=color, label="Transmission")
plt.plot(freq, out.best_fit, '--k', label="Fit")
# plt.xlim(xlim)
plt.grid(True)
plt.legend(shadow=True)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Transmission (A.U.)")

plt.tight_layout()
plt.show()


# do fitting of off-resonant
ramp = df_off['CH1'].astype(float).to_numpy()
transmission = df_off['CH2'].astype(float).to_numpy()

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
ramp = ramp[id_min:id_max]
transmission = transmission[id_min:id_max]

# convert time to frequency
freq = np.linspace(0, (FREQ_OFF[1] - FREQ_OFF[0]), id_max-id_min)  # unit: GHz

model = ConstantModel() + BreitWignerModel()
out = model.fit(transmission, x=freq,
                center=4.475, amplitude=0.2, q=0,
                c=0.1, sigma=0.4)

# print out relevant information
print("Off-resonant data:")
sigma = out.params['sigma'].value  # unit: GHz
sigma_err = out.params['sigma'].stderr
freq_light = FREQ_OFF[0] + out.params['center'].value  # unit: GHz
freq_err = np.sqrt(FREQ_ERR**2 + out.params['center'].stderr**2)
q_off = freq_light / sigma
q_off_err = q_off * np.sqrt((freq_err/freq_light)**2 + (sigma_err/sigma)**2)
print(f"\tSigma: {sigma:.6f} +/- {sigma_err:.6f} GHz")
print(f"\tCavity freq: {freq_light:.6f} +/- {freq_err:.6f} GHz")
print(f"\tQ: {q_off:.0f} +/- {q_off_err:0.0f}")

# plotting
plt.plot(freq, transmission, color=color, label="Transmission")
plt.plot(freq, out.best_fit, '--k', label="Fit")
# plt.xlim(xlim)
plt.grid(True)
plt.legend(shadow=True)
plt.xlabel("Frequency (GHz)")
plt.ylabel("Transmission (A.U.)")

plt.tight_layout()
plt.show()


# final result
diff = q_on - q_off
diff_err = np.sqrt(q_on_err**2 + q_off_err**2)
print("Difference:")
print(f"\tQ_on - Q_off: {diff:.0f} +/- {diff_err:.0f}")
