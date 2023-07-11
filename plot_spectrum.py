import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


SIGNAL_DATA = "/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab members/ZhongLab_Alex" \
              "/Entangled Photon Measurements/Spectrum Measurements/spectrometer_experiment_p1_highres.csv"
IDLER_DATA = "/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab members/ZhongLab_Alex" \
             "/Entangled Photon Measurements/Spectrum Measurements/spectrometer_experiment_p2_highres.csv"
OUTPUT = "output_figs/spectrum"

# plotting parameters
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
kw = {'height_ratios': [10, 3],
      'width_ratios': [3, 10]}
figsize = (8, 6)


# get data for photons
column_names = ['wavelength', 'counts']
df_signal = pd.read_csv(SIGNAL_DATA, names=column_names)
df_idler = pd.read_csv(IDLER_DATA, names=column_names)

# spectrum = np.arange(1530, 1540.1, 0.5)
# signal = [140, 150, 140, 180, 170, 190, 210, 220, 300, 350, 260, 210, 200, 230, 180, 150, 150, 120, 160, 130, 150]
# idler = [220, 200, 230, 250, 230, 280, 270, 320, 350, 370, 360, 300, 300, 260, 230, 200, 220, 180, 200, 200, 210]
#
# assert len(signal) == len(idler) == len(spectrum)


fig, ax = plt.subplots()
ax.plot(df_signal['wavelength'].astype(float), df_signal['counts'].astype(int),
        '-o', color='cornflowerblue',
        label="Signal Photon")
ax.plot(df_idler['wavelength'].astype(float), df_idler['counts'].astype(int),
        '-o', color='coral',
        label="Idler Photon")

ax.set_title("Spectrum Data")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Coincidence Counts")
ax.legend(shadow=True)

fig.tight_layout()
fig.savefig(OUTPUT)
