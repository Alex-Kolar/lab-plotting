import glob
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit import Model
from lmfit.models import ConstantModel, BreitWignerModel

from ring_resonators.cavity_fit.cavity_metrics import *


# resonance data used to predict pair rate
CAVITY_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
               "/New_mounted_device/10mK/03262025/SDS00003.csv")
SCAN_RANGE = (194805.651, 194814.283)

# parameters for the device (to use for pair rate calculation)
n_eff = 2.18
L = np.pi * 220e-6  # unit: m
c = 3e8  # unit: m/s
gamma = (0.0028807694600181634, 0.00044581372164523476)  # two solutions

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
color_coincidence = 'coral'


# gather cavity data
cavity_df = pd.read_csv(CAVITY_DATA, header=10, skiprows=[11])
ramp = cavity_df['CH1'].astype(float)
transmission = cavity_df['CH2'].astype(float)

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
transmission = transmission[id_min:id_max]
transmission.reset_index(drop=True, inplace=True)
freq = np.linspace(0, SCAN_RANGE[1] - SCAN_RANGE[0],
                   num=(id_max-id_min))  # unit: MHz

# do fitting (and determine guesses for fit)
max_trans = max(transmission)
model = ConstantModel() + BreitWignerModel()
res = model.fit(transmission, x=freq,
                c=0.4,
                amplitude=0.7,
                center=4.4,
                sigma=0.6,
                q=-0.1)
print("\nResonance fitting:")
print(res.fit_report())


# plot cavity fitting
plt.plot(freq, transmission,
         color='cornflowerblue', label='Data')
plt.plot(freq, res.best_fit,
         ls='--', color='k', label='Fit')
# plt.plot(freq, res.init_fit,
#          ls='--', color='r', label='Initial Guess')

plt.title('Coincidence Pump Resonance')
plt.xlabel(f'Detuning from {SCAN_RANGE[0]:.3f} (GHz)')
plt.ylabel('Transmission (A.U.)')
plt.legend(shadow=True)

plt.tight_layout()
plt.show()


# get relevant info from cavity fit
width = res.params[f'sigma'].value  # unit: GHz
center = res.params[f'center'].value  # unit: GHz
amplitude = res.params[f'amplitude'].value
constant = res.params[f'c'].value

freq_light = SCAN_RANGE[0] + center  # unit: GHz
q = freq_light / width
contrast = amplitude / (amplitude + constant)
print("\n")
print(f"Cavity kappa: {width}")
print(f"Cavity Q: {q}")
print(f"Cavity contrast: {contrast}")

# calculate enhancement
enhance_1, enhance_2 = calculate_enhancement(freq_light*1e-3, q, contrast,
                                             L, n_eff)
print("\n")
print("Field enhancement 1:", np.sqrt(enhance_1))
print("Field enhancement 2:", np.sqrt(enhance_2))


# calculate rates
rate_1, rate_2 = calculate_rates(freq_light*1e-3, q, contrast,
                                 L, n_eff, gamma[0])
rate_3, rate_4 = calculate_rates(freq_light*1e-3, q, contrast,
                                 L, n_eff, gamma[1])
print("\n")
print("Possible Rates:")
for rate in (rate_1, rate_2, rate_3, rate_4):
    print(rate)
