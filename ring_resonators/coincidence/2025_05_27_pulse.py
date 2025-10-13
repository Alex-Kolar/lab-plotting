import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, ConstantModel



FILENAME = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/10mK/coincidence_2025_05_27/SDS00001.csv")

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'

# fitting params
FITTING = True
fit_range = (-100, 150)


# read data
df = pd.read_csv(FILENAME, header=10, skiprows=[11])
time = df['Source'].astype(float).to_numpy()
pulse = df['CH2'].astype(float).to_numpy()
time *= 1e9  # convert to ns

# fitting
if FITTING:
    idx_to_fit = np.where(np.logical_and(time > fit_range[0], time < fit_range[1]))[0]
    time_to_fit = time[idx_to_fit]
    pulse_to_fit = pulse[idx_to_fit]
    model = GaussianModel() + ConstantModel()
    res = model.fit(pulse_to_fit, x=time_to_fit,
                    center=20, sigma=20, c=-0.1, amplitude=100)
    print(res.fit_report())


# plotting
if FITTING:
    plt.plot(time, pulse, color=color,
             label='Data')
    plt.plot(time_to_fit, res.best_fit,
             color='k', ls='--',
             label='Fit')
else:
    plt.plot(time, pulse, color=color)

plt.title('Pump Pulse Shape')
plt.xlabel('Time (ns)')
plt.ylabel('Pulse Amplitude (A.U.)')
plt.legend(shadow=True)
plt.xlim(-100, 200)

plt.tight_layout()
plt.show()
