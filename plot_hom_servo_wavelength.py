import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from lmfit.models import Model


# fitting parameters
def triangle_fit(x, bckgd, x0, width, height):
    shift_x = x - x0
    triangle = height * (1 - (2 * np.abs(shift_x) / width))
    triangle[np.abs(shift_x) >= width/2] = 0
    triangle += bckgd
    return triangle


def gauss_fit(x, bckgd, x0, sigma, height):
    gauss = height * np.exp((-(x - x0) ** 2) / (2 * sigma ** 2))
    gauss = gauss + bckgd
    return gauss


# experimental and CSV parameters
data_filename = \
    '/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab members/ZhongLab_Alex' \
    '/Entangled Photon Measurements/spec_hom_optimize_458mW_25Apr2023.csv'
WAVELENGTH = 'Wavelength'
SERVO_POS = 'Servo_pos'
MAX_COUNT = 'Max_count'
NOISE = 'noise'
SINGLE_1 = 'raw counts 7'
SINGLE_2 = 'raw counts 8'

# plotting parameters
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
kw = {'height_ratios': [3, 10],
      'width_ratios': [3, 10]}
figsize = (10, 8)
x_range = (42, 47)  # unit: ps
y_range = (0, 1600)

# output filenames
OUTPUT_DIR = 'output_figs'
FILENAME_COINCIDENCE = 'coincidences.png'
FILENAME_SINGLES = 'singles.png'

_, tail = os.path.split(data_filename)
output_subdir = os.path.splitext(tail)[0]
output_dir = os.path.join(OUTPUT_DIR, output_subdir)
try:
    os.makedirs(output_dir)
    print(f'Created directory {output_dir}')
except FileExistsError:
    pass

# wavelength to analyze and fit
WAVELENGTH_TO_FIT = 1535.5  # unit: nm


df = pd.read_csv(data_filename)

# figure out how many entries per wavelength
wavelengths_all = df[WAVELENGTH].to_numpy()
wavelengths = np.unique(wavelengths_all)
num_per_wl = len(wavelengths_all) // len(wavelengths)
print("Number of unique wavelengths:", len(wavelengths))
print("Number of entries per wavelength:", num_per_wl)

# get servo positions
servo_pos = df[SERVO_POS][:num_per_wl].to_numpy()

# get coincidence data and singles data
coincidences_series = df[MAX_COUNT]
coincidences = np.resize(coincidences_series, (len(wavelengths), num_per_wl))
coincidences = coincidences.astype(float)

singles_1_series = df[SINGLE_1]
singles_1 = np.resize(singles_1_series, (len(wavelengths), num_per_wl))
singles_1 = singles_1.astype(float)
singles_2_series = df[SINGLE_2]
singles_2 = np.resize(singles_2_series, (len(wavelengths), num_per_wl))
singles_2 = singles_2.astype(float)

# subtract noise
noise_series = df[NOISE]
noise = np.resize(noise_series, (len(wavelengths), num_per_wl))
coincidences_adj = coincidences - noise

# do integration
coincidences_servo = np.sum(coincidences_adj, axis=0)
coincidences_wavelength = np.sum(coincidences_adj, axis=1)

# plotting
fig, ax = plt.subplots(2, 2, gridspec_kw=kw, figsize=figsize)
ax[0][1].sharex(ax[1][1])
ax[1][0].sharey(ax[1][1])
fig.delaxes(ax[0][0])

X, Y = np.meshgrid(servo_pos, wavelengths)
pcm = ax[1][1].pcolormesh(X, Y, coincidences_adj, cmap='magma')
cb = fig.colorbar(pcm, ax=ax[1][1], label="Coincidence Counts")
ax[0][1].plot(servo_pos, coincidences_servo)
ax[1][0].plot(coincidences_wavelength, wavelengths)

ax[1][1].tick_params(labelleft=False)
ax[0][1].tick_params(labelbottom=False,
                     labelleft=False)
ax[1][0].tick_params(labelbottom=False)
ax[1][1].set_xlabel("Servo Position (mm)")
ax[1][0].set_ylabel("Wavelength (nm)")

fig.tight_layout()

# do axis adjusting
pos = ax[1][1].get_position()
pos_upper = ax[0][1].get_position()
print(pos)
print(pos_upper)
delta = pos_upper.width - pos.width
# new = [pos.x0, pos.y0, pos_upper.width, pos.height]
# ax.set_position(new)
# pos_c = cb.ax.get_position()
# new = [pos_c.x0 + delta, pos_c.y0, pos_c.width, pos_c.height]
# cb.ax.set_position(new)

fig.savefig(os.path.join(output_dir, FILENAME_COINCIDENCE))
print("Finished generating 2D plot.")
plt.close()

# # plotting of singles counts
# plt.pcolormesh(X, Y, singles_1)
# plt.colorbar(label="Counts (Channel 1)")
# plt.xlabel("Servo Position (mm)")
# plt.ylabel("Wavelength (nm)")
#
# plt.tight_layout()
# plt.show()
# plt.close()
#
# plt.pcolormesh(X, Y, singles_2)
# plt.colorbar(label="Counts (Channel 2)")
# plt.xlabel("Servo Position (mm)")
# plt.ylabel("Wavelength (nm)")
#
# plt.tight_layout()
# plt.show()
# plt.close()


# # do some fitting
# wave_diff = np.abs(wavelengths - WAVELENGTH_TO_FIT)
# wave_idx = wave_diff.argmin()
# print(f"Closest wavelength: {wavelengths[wave_idx]}")
#
# # coincidence_individual = coincidences[wave_idx]
# # model = Model(gauss_fit)
# # res = model.fit(coincidence_individual, x=servo_pos)
# # res.fit_report()
#
# plt.plot(servo_pos, coincidences[wave_idx],
#          'o', label="Coincidence Counts")
# plt.plot(servo_pos, noise[wave_idx],
#          'o', label="False Counts")
# plt.title(f"{WAVELENGTH_TO_FIT} nm")
# plt.legend(shadow=True)
# plt.show()
