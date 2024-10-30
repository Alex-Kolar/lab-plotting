import pandas as pd
from lmfit.models import GaussianModel, LinearModel
import matplotlib as mpl
import matplotlib.pyplot as plt


DATA_FILE = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Lithium Niobate"
             "/Absorption Scan/Chip 1/cln_12212023/absorption_200avg_2.csv")
REF_FILE = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Lithium Niobate"
            "/Absorption Scan/Chip 1/cln_12212023/laser_ref_linear_200avg.csv")
CSV_HEAD = ["wavelength", "power"]

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'

# fitting params
wl_fit_range = (1527, 15)


# get data
df_data = pd.read_csv(DATA_FILE, names=CSV_HEAD)
df_ref = pd.read_csv(REF_FILE, names=CSV_HEAD)

wavelength = df_data['wavelength']
power_norm = df_data['power'] / df_ref['power']

# do fitting
model = GaussianModel() + LinearModel()
idx_to_fit = (wavelength > wl_fit_range[0]) & (wavelength < wl_fit_range[1])
wl_to_fit = wavelength[idx_to_fit]
power_to_fit = 1 - power_norm[idx_to_fit]
res = model.fit(power_to_fit, x=wl_to_fit,
                center=1532)
print("Gaussian hole fitting results:")
print(res.fit_report())

# do conversions
center_wl = res.params['center'].value
center_wl_err = res.params['center'].stderr
fwhm_wl = res.params['fwhm'].value

center_freq = 3e8 / (center_wl * 1e-9)  # units: Hz
center_freq *= 1e-9  # units: GHz
freq_upper = 3e8 / ((center_wl - (fwhm_wl/2)) * 1e-9)  # units: Hz
freq_lower = 3e8 / ((center_wl + (fwhm_wl/2)) * 1e-9)  # units: Hz
fwhm_freq = (freq_upper - freq_lower) * 1e-9  # units: GHz

print("\nMain results:")
print(f"\tCenter: {center_wl} +/- {center_wl_err} nm")
print(f"\tCenter: {center_freq} GHz")
print(f"\tFWHM: {fwhm_freq} GHz")


plt.plot(wavelength, power_norm,
         color=color, label='Data')
plt.plot(wl_to_fit, 1-res.best_fit,
         '--k', label='Fit')
plt.axvspan(*wl_fit_range,
            alpha=0.2, color='gray', label='Fit Region')

plt.title("Er:CLN (Chip 1) Transmission Scan")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Normalized Power")
plt.legend(shadow=True)
plt.grid('on')

plt.tight_layout()
plt.show()
