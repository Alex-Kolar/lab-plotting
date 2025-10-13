import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


FILENAME_PULSE = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
                  "/New_mounted_device/10mK/06052025/SDS00001.csv")
rep_freq = 5e3  # unit: Hz
expected_rate = 1122  # unit: pairs/s/mW^2
device_eff = 0.056  # per face
snspd_eff = 0.7
measured_afc_eff = 0.0002


# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
xlim_range = 20  # size of x lims
fit_range = 500  # size for fitting
color = 'coral'
color_pulse = 'cornflowerblue'


# eta equation
def eta_comb_sq(F):
    """For square comb teeth.

    Note that np.sinc(x) evaluates to sin(pi*x)/(pi*x).
    """
    return np.square(np.sinc(1 / F))


# get pulse data
df_pulse = pd.read_csv(FILENAME_PULSE, header=10, skiprows=[11])
pulse = df_pulse["CH1"].to_numpy()  # unit: W (10^2 V/W, 20 dB attenuation)
time_pulse = df_pulse["Source"].to_numpy()  # unit: s
time_pulse *= 1e9  # unit: ns

# integrate squared pulse power
time_step = time_pulse[1] - time_pulse[0]  # unit: ns
time_len = time_pulse[-1] - time_pulse[0]  # unit: ns
avg_sq_power = np.sum(np.multiply(pulse, pulse)) * time_step / time_len  # unit: W^2
avg_sq_power *= 1e6  # unit: mW^2

duty_cycle_measure = (time_len * 1e-9) * (rep_freq)
avg_sq_power *= duty_cycle_measure
avg_sq_power_on_chip = avg_sq_power * (device_eff ** 2)

print("average square power off-chip (mW^2):", avg_sq_power)
print("average square power on-chip (mW^2):", avg_sq_power_on_chip)
print("expected pair rate on-chip (/s):", avg_sq_power_on_chip * expected_rate)
print("expected pair rate measured (/s):", avg_sq_power_on_chip * expected_rate * (device_eff ** 2) * (snspd_eff ** 2))




