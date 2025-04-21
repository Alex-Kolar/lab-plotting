import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit import Model


FILENAME = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/300K_no_erbium/coincidence/01102025"
            "/Correlation_10min_100psbw_device24_195011GHz.txt")
FITTING = True  # add a fit
WAVELENGTH = 1537.782  # units: nm
SAVE_FILE = False
OUTPUT_PATH = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs"
               "/ring_resonators/original/coincidence_100ps_mod_fit.pdf")

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 10})
xlim_range = 15  # size of x lims
color = 'coral'


def g_2_no_delta(x, x0, amplitude, kappa, g):
    x = np.abs(x - x0)
    exp_term = g*np.sinh(g*x) + (kappa/2)*np.cosh(g*x)
    g_2 = 1 + (np.exp(-kappa * x) / (g ** 2)) * (np.abs(exp_term) ** 2)
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

    x0_guess = 30
    amplitude_guess = 80
    kappa_guess = 1
    g_guess = 0.125

    res = model.fit(coincidence, x=time,
                    x0=x0_guess,
                    amplitude=amplitude_guess,
                    kappa=kappa_guess,
                    g=g_guess)
    print(res.fit_report())


# plotting
fig, ax = plt.subplots(figsize=(4, 2), dpi=400)

ax.bar(time, coincidence, width=time_diff, color=color, label='Data')
if FITTING:
    time_to_plot = np.linspace(min(time), max(time), 10000)
    ax.plot(time_to_plot, g_2_no_delta(time_to_plot, **res.best_values),
            'k--', label='Fit')

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

    print('k:', kappa)
    print('correlation:', 1/kappa)

center = res.params['x0'].value
xlim = (center - xlim_range/2, center + xlim_range/2)
ax.set_xlim(xlim)
ax.set_xlabel("Timing Offset (ns)")
ax.set_ylabel("Coincidence Counts")
ax.set_title("Two-Photon Coincidence")
plt.legend(shadow=True)

fig.tight_layout()
if SAVE_FILE:
    fig.savefig(OUTPUT_PATH)
else:
    fig.show()

# if FITTING:
#     cav_freq = 3e8 / (WAVELENGTH * 1e-9)  # unit: Hz
#     cav_freq *= 1e-6  # unit: MHz
#     Q = cav_freq / kappa
#     print("Measured Q: {}".format(Q))
#
#     g = res.params['g'].value  # unit: 2*pi*GHz
#     g /= 2 * np.pi
#     g *= 1e3  # unit: MHz
#     print("Measured g: {} MHz".format(g))
#
#
#     spec_line = 1 / ((sigma_single * 1e-9) * 2 * np.pi)  # unit: Hz
#     spec_line *= 1e-6  # unit: MHz
#     cav_freq = 3e8 / (WAVELENGTH * 1e-9)  # unit: Hz
#     cav_freq *= 1e-6  # unit: MHz
#     print("Sigma (single): {} ns".format(sigma_single))
#     print("Sigma (frequency): {} MHz".format(spec_line))
#     print("Q: {}".format(cav_freq / spec_line))

# plotting of wider view w/o fit
fig, ax = plt.subplots()

ax.bar(time, coincidence, width=time_diff, color=color, label='Data')

center = res.params['x0'].value
xlim = (center - xlim_range, center + xlim_range)
ax.set_xlim(xlim)
ax.set_xlabel("Timing Offset (ns)")
ax.set_ylabel("Coincidence Counts")
ax.set_title("Two-Photon Coincidence")

fig.tight_layout()
fig.show()

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
