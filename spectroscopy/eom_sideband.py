import pandas as pd
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel

LASER_REF = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data"
             "/Er YSO Spectroscopy/eom_sidebands/laser_ref.csv")
SIDEBANDS = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data"
             "/Er YSO Spectroscopy/eom_sidebands/sideband_2GHz.csv")
OSA_HEADER = ["Wavelength (nm)", "Power (mW)"]
RF_FREQ = 2e9  # units: Hz
c = 3e8  # units: m/s

laser_ref_df = pd.read_csv(LASER_REF, names=OSA_HEADER)
sidebands = pd.read_csv(SIDEBANDS, names=OSA_HEADER)

wl = laser_ref_df["Wavelength (nm)"]
laser_ref = laser_ref_df["Power (mW)"]
mod = sidebands["Power (mW)"]

# fit laser peak
model = GaussianModel()
res = model.fit(laser_ref, x=wl,
                center=1534.96, sigma=0.01)
center = res.params["center"].value

# get upper and lower expected bands
freq_center = c / (center * 1e-9)
freq_upper = freq_center + RF_FREQ
freq_lower = freq_center - RF_FREQ
wl_upper = (c / (freq_upper)) * 1e9
wl_lower = (c / (freq_lower)) * 1e9
print(wl_upper, wl_lower)

plt.plot(wl, laser_ref,
         color='cornflowerblue', label="RF off")
plt.plot(wl, mod,
         color='coral', label="RF on")
plt.axvline(center, color='k', ls='--', label="Laser Center")
plt.axvline(wl_upper, color='gray', ls='--', label=r"$\pm$2 GHz")
plt.axvline(wl_lower, color='gray', ls='--')
plt.xlim((1534.89, 1535.02))
plt.title("Sideband Generation")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Power (mW)")
plt.legend(shadow=True)
plt.grid(True)
plt.show()
