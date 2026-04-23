import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, ConstantModel


FILENAME = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
            '/Mounted_device_mk_5/10mK/2026_03_05/pair_storage/pair_storage_test7.txt')
FILENAME_OFFRES = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
                   '/Mounted_device_mk_5/10mK/2026_03_05/pair_storage/off_res_1.txt')


# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
xlim_range = 4.5  # size of x lims (in us)
xlim_range_small = 0.1
ylim = (0, 800)
color_offres = 'cornflowerblue'
color = 'coral'


# NOTE: first line of csv needs extra tab added
# otherwise the columns will not be read properly
df = pd.read_csv(FILENAME, sep='\t')
df_offres = pd.read_csv(FILENAME_OFFRES, sep='\t')
coincidence = df["Counts"]
coincidence_offres = df_offres["Counts"]
time = df["Time(ps)"]  # unit: ps
time *= 1e-6  # unit: us
time_diff = time[1] - time[0]  # spacing of histogram

center = time[np.argmax(coincidence)]
time -= center


# plot overlaid
fig, ax = plt.subplots(figsize=(6, 4), dpi=400)
plt.plot(time, coincidence_offres, color=color_offres,
         label='Off-Resonant Signal')
plt.plot(time, coincidence, color=color,
         label='On-Resonant Signal')
plt.xlim(-xlim_range/2, xlim_range/2)
plt.ylim(ylim)
plt.xlabel(r"Timing Offset ($\mathrm{\mu}$s)")
plt.ylabel("Coincidence Counts")
plt.title("Two-Photon Coincidence with AFC Storage")
plt.legend()
plt.tight_layout()
plt.show()

# plot zoomed-in input coincidence (no echo)
fig, ax = plt.subplots(figsize=(3, 4), dpi=400)

ax.plot(time, coincidence_offres, color=color_offres)

ax.set_xlim(-xlim_range_small/2, xlim_range_small/2)
ax.set_ylim(ylim)
ax.set_xlabel(r"Timing Offset ($\mathrm{\mu}$s)")
ax.set_ylabel("Coincidence Counts")
ax.set_title("Input")

fig.tight_layout()
fig.show()

# plot zoomed-in echo coincidence (no input)
fig, ax = plt.subplots(figsize=(3, 4), dpi=400)

ax.plot(time, coincidence, color=color)

ax.set_xlim(-xlim_range_small/2+1, xlim_range_small/2+1)
ax.set_ylim(0, 40)
ax.set_xlabel(r"Timing Offset ($\mathrm{\mu}$s)")
ax.set_ylabel("Coincidence Counts")
ax.set_title("Echo")

fig.tight_layout()
fig.show()
