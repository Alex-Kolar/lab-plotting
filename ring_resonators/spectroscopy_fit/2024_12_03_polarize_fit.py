import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import VoigtModel, ConstantModel


# collected data
BG = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
      "/Bulk_crystal/10mK/12032024/LASER_OFF.csv")
DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
        "/Bulk_crystal/10mK/12032024/AFTERBURN.csv")
START_FREQ = 194810.216  # unit: GHz
END_FREQ = 194818.918  # unit: GHz

# fit params
fit_start = 3.3
fit_end = 3.9

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
color_od = 'coral'


# reference level
df_laser_off = pd.read_csv(BG, header=10, skiprows=[11])
off_level = np.mean(df_laser_off['CH2'].astype(float).to_numpy())

# collected data
df = pd.read_csv(DATA, header=10, skiprows=[11])

ramp = df['CH1'].astype(float).to_numpy()
transmission = df['CH2'].astype(float).to_numpy()
transmission -= off_level

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
ramp = ramp[id_min:id_max]
transmission = transmission[id_min:id_max]

# convert to optical depth
bg = max(transmission)
od = np.log(bg / transmission)

# convert time to frequency
freq = np.linspace(0, (END_FREQ - START_FREQ), id_max - id_min)  # unit: GHz

# do fitting
freq_bounds = np.logical_and(freq > fit_start, freq < fit_end)
real_bounds = np.logical_and(freq_bounds, ~np.isnan(od))
idx_to_fit = np.where(real_bounds)[0]
model = VoigtModel() + ConstantModel()
res = model.fit(od[idx_to_fit], x=freq[idx_to_fit],
                center=3.55, sigma=0.25, c=0.2)
print(res.fit_report())


# plot optical depth and pumping
plt.plot(freq, od,
         color=color_od, label="Data")
plt.plot(freq[idx_to_fit], res.best_fit,
         color='k', ls='--', label="Fit")

# add label for FWHM
horizontal_pos = 3.7
vertical_pos = 2.5
text = rf"Height: OD {res.params['height'].value:.3f} $\pm$ {res.params['height'].stderr:.3f}"
text += "\n"
text += rf"FWHM: {res.params['fwhm'].value * 1e3:.3f} $\pm$ {res.params['fwhm'].stderr * 1e3:.3f} MHz"
plt.text(horizontal_pos, vertical_pos, text,
         ha='left', va='center')

plt.title("Hyperfine Polarization Fit")
plt.xlabel(f"Frequency + {START_FREQ:.3f} (GHz)")
plt.ylabel("Optical Depth")
plt.legend(shadow=True)
plt.xlim((3, 4.5))
plt.ylim((0, 4))

plt.tight_layout()
plt.show()
