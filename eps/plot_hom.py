import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# fitting parameters
def triangle_fit(x, bckgd, x_center, width, height):
    shift_x = x - x_center
    triangle = height * (1 - (2 * np.abs(shift_x) / width))
    triangle[np.abs(shift_x) >= width/2] = 0
    triangle += bckgd
    return triangle


def gauss_fit(x, bckgd, x0, sigma, height):
    gauss = height * np.exp((-(x - x0) ** 2) / (2 * sigma ** 2))
    gauss = gauss + bckgd
    return gauss


# experimental parameters
data_filename = 'data/hom_3_02/hom_experiment_5_8_mm.csv'
num_bins = 1000
servo_pos = np.arange(5, 8, 0.005)
integration_time = 10  # unit: s

# plotting parameters
output_filename = 'output_figs/hom_5_8.png'
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
x_range = (42, 47)  # unit: ps
y_range = (0, 1600)


columns = ["{}".format(ns) for ns in range(num_bins)]
df = pd.read_csv(data_filename, names=columns)

# get bin for max coincidence counts
row = df.iloc[0]
bin_max = row.idxmax()
coincidence = df[bin_max].to_numpy()
coincidence = coincidence.astype(float)
coincidence = coincidence.flatten()

# get bin for false coincidence
time_offset = 20  # unit: ns (from paper)
bin_false = str(int(bin_max) + time_offset)
coincidence_false = df[bin_false].to_numpy()
coincidence_false = coincidence_false.astype(float)
coincidence_false = coincidence_false.flatten()

# convert displacement to timing
relative_timing = (servo_pos * 1e-3) / 3e8
relative_timing *= 1e12  # convert to ps
relative_timing *= 2  # multiply by 2, since servo offset is doubled in path length

# fitting params
fit_range = x_range
idx_to_keep = (relative_timing >= fit_range[0]) & (relative_timing <= fit_range[1])
coincidence_short = coincidence[idx_to_keep]
timing_short = relative_timing[idx_to_keep]
bckgd_guess = np.max(coincidence_short)
height_guess = np.min(coincidence_short) - bckgd_guess
center_guess = timing_short[np.argmin(coincidence_short)]

# fitting (scipy)
res = curve_fit(triangle_fit, timing_short, coincidence_short,
                p0=[bckgd_guess, center_guess, 1, height_guess])
popt = res[0]

# calculate visibility
c_max = popt[0] - np.mean(coincidence_false)
c_min = c_max + popt[3] - np.mean(coincidence_false)
vis = (c_max - c_min) / (c_max + c_min)

# calculate photon width
triangle_width = popt[2]
photon_width = triangle_width / 2

min_idx = np.argmin(df[bin_max])
min_time = popt[1]
min_time *= 1e-12  # convert to s
min_pos = min_time * 3e8 * 1e3  # unit: mm
min_pos /= 2  # double path length
print("Position of minimum: {} mm".format(servo_pos[min_idx]))
print("Position of minimum (fitted): {} mm".format(min_pos))
print("Visibility: {}%".format(vis * 100))
print("Photon Width: {} ps".format(photon_width))


# plotting
fig, ax = plt.subplots()
ax.plot(relative_timing, coincidence, 'o', color='cornflowerblue',
        label='Coincidences')
ax.plot(relative_timing, triangle_fit(relative_timing, *popt), '--k',
        label='Best fit')
ax.plot(relative_timing, coincidence_false, 'o', color='coral',
        label='False Coincidences')

ax.set_xlim(x_range)
ax.set_ylim(y_range)
ax.set_title("HOM Coincidence Dip")
ax.set_xlabel("Timing Offset (ps)")
ax.set_ylabel("Coincidence Counts ({} seconds)".format(integration_time))
ax.legend(shadow=True)

result_text = "Visibility: {:.1f}%".format(vis * 100)
result_text += "\n"
result_text += "Photon width: {:.3f} ps".format(photon_width)
ax.text(45.25, 100, result_text)

fig.tight_layout()
plt.savefig(output_filename)
# fig.show()
