import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit import Model


FILENAME = "/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators" \
           "/Original_device/Coincidence Count Measurement/08022023/Correlation-2_2023-08-03_14-43-39_(30sec_int).txt"
FITTING = True  # add a fit
WAVELENGTH = 1537.782  # units: nm
SAVE_FILE = False
OUTPUT_PATH = "/output_figs/ring_resonators/original/coincidence_1ns.svg"

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Arial',
                     'font.size': 12})
xlim = (-7, 7)
color = 'coral'


def g_2_no_delta(x, x0, amplitude, kappa, g):
    x = np.abs(x - x0)
    exp_term = g*np.sinh(g*x) + (kappa/2)*np.cosh(g*x)
    g_2 = 1 + (np.exp(-kappa * x) / (g ** 2)) * (exp_term ** 2)
    return amplitude * g_2


# NOTE: first line of csv needs extra tab added
# otherwise the columns will not be read properly
df = pd.read_csv(FILENAME, sep='\t')
coincidence = df["Counts"]
time = df["Time(ps)"]  # unit: ps
time *= 1e-3  # unit: ns
time_diff = time[1] - time[0]  # spacing of histogram

# fitting
if FITTING:
    model = Model(g_2_no_delta)

    x0_guess = 0
    amplitude_guess = 40
    kappa_guess = 2
    g_guess = 0.5

    res = model.fit(df["Counts"], x=time,
                    x0=x0_guess,
                    amplitude=amplitude_guess,
                    kappa=kappa_guess,
                    g=g_guess)
    print(res.fit_report())


# plotting
fig, ax = plt.subplots()

ax.bar(time, coincidence, width=time_diff, color=color)
if FITTING:
    ax.plot(time, res.best_fit, 'k--')

    # extract relevant info
    kappa = res.params['kappa'].value  # unit: 2*pi*GHz
    kappa /= 2*np.pi
    kappa *= 1e3  # unit: MHz
    kappa_err = res.params['kappa'].stderr  # unit: GHz
    kappa_err /= 2*np.pi
    kappa_err *= 1e3  # unit: MHz

    label = r"$\kappa$: {:0.3f} $\pm$ {:0.3f} MHz".format(
        kappa, kappa_err)
    t = ax.text(0.05, 0.95, label,
                horizontalalignment='left', verticalalignment='top')
    t.set_transform(ax.transAxes)

ax.set_xlim(xlim)
ax.set_xlabel("Timing Offset (ns)")
ax.set_ylabel("Coincidence Counts")
ax.set_title("100 ps Bin Width")

fig.tight_layout()
if SAVE_FILE:
    fig.savefig(OUTPUT_PATH)
else:
    fig.show()

if FITTING:
    cav_freq = 3e8 / (WAVELENGTH * 1e-9)  # unit: Hz
    cav_freq *= 1e-6  # unit: MHz
    Q = cav_freq / kappa
    print("Measured Q: {}".format(Q))

    g = res.params['g'].value  # unit: 2*pi*GHz
    g /= 2 * np.pi
    g *= 1e3  # unit: MHz
    print("Measured g: {} MHz".format(g))


    # spec_line = 1 / ((sigma_single * 1e-9) * 2 * np.pi)  # unit: Hz
    # spec_line *= 1e-6  # unit: MHz
    # cav_freq = 3e8 / (WAVELENGTH * 1e-9)  # unit: Hz
    # cav_freq *= 1e-6  # unit: MHz
    # print("Sigma (single): {} ns".format(sigma_single))
    # print("Sigma (frequency): {} MHz".format(spec_line))
    # print("Q: {}".format(cav_freq / spec_line))

# determine coincidence rate
center = coincidence.idxmax()
bin_range = 50  # units: 100ps
size = []
all_counts = []
all_car = []
for i in range(bin_range):
    lower = center - i
    upper = center + i
    total_counts = np.sum(coincidence[lower:(upper+1)])

    num_bins = (2*i) + 1
    size.append(num_bins * 0.1)

    # calculate count rate
    counts_per_sec = total_counts / 30
    all_counts.append(counts_per_sec)

    # determine car
    avg_bg_level = res.params['amplitude'].value  # counts per 100 ps bin
    bg_counts = avg_bg_level * num_bins
    car = (total_counts - bg_counts) / bg_counts
    all_car.append(car)


# plotting
color_counts = 'cornflowerblue'
color_car = 'coral'

fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(size, all_counts,
        'o-', color=color_counts)
ax2.plot(size, all_car,
         'o-', color=color_car)
ax.set_xlabel("Bin size (ns)")
ax.set_ylabel("Coincidence Counts (pairs/s)",
              color=color_counts)
ax2.set_ylabel("CAR",
               color=color_car)
ax.grid(True)

fig.tight_layout()
fig.show()

for pairs, car in zip(all_counts, all_car):
    print(pairs, car)

# num_bins = 10
# center = coincidence.idxmax()
# lower = center - 5
# upper = center + 5
# counts_1_ns = np.sum(coincidence[lower:upper])
# counts_per_sec = counts_1_ns / 30
# print("Counts per second: {}".format(counts_per_sec))
