import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import LorentzianModel, ConstantModel
from scipy import fft


FILENAME = "/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators" \
           "/Original_device/Coincidence Count Measurement/08022023/Correlation-2_2023-08-03_14-41-25_(30sec_int).txt"
FITTING = False  # add a gaussian fit
WAVELENGTH = 1537.782  # units: nm
SAVE_FILE = True
OUTPUT_PATH = "/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators/original/coincidence_1ns.svg"

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Arial',
                     'font.size': 12})
xlim = (-50, 50)
color = 'cornflowerblue'

# NOTE: first line of csv needs extra tab added
# otherwise the columns will not be read properly
df = pd.read_csv(FILENAME, sep='\t')
print(df)
coincidence = df["Counts"]
time = df["Time(ps)"]  # unit: ps
time *= 1e-3  # unit: ns
time_diff = time[1] - time[0]  # spacing of histogram

# fitting
if FITTING:
    model = LorentzianModel() + ConstantModel()
    center_guess = time[coincidence.idxmax()]
    amplitude_guess = coincidence.max()
    res = model.fit(df["Counts"], x=time,
                    center=center_guess, amplitude=amplitude_guess)
    print(res.fit_report())

    sigma = res.params['sigma'].value
    sigma_single = sigma/np.sqrt(2)
    fwhm_single = 2*sigma_single


# plotting
fig, ax = plt.subplots()

ax.bar(time, coincidence, width=time_diff, color=color)
if FITTING:
    ax.plot(time, res.best_fit, 'k--')
    label = r"FWHM: {:0.3f} $\pm$ {:0.3f} ns".format(
        2*res.params['sigma'].value, 2*res.params['sigma'].stderr)
    label += "\n"
    label += r"FWHM (single): {:0.3f} ns".format(
        fwhm_single)
    t = ax.text(0.05, 0.95, label,
                horizontalalignment='left', verticalalignment='top')
    t.set_transform(ax.transAxes)

ax.set_xlim(xlim)
ax.set_xlabel("Timing Offset (ns)")
ax.set_ylabel("Coincidence Counts")
ax.set_title("1 ns Bin Width")

fig.tight_layout()
if SAVE_FILE:
    fig.savefig(OUTPUT_PATH)
else:
    fig.show()

if FITTING:
    spec_line = 1 / ((sigma_single * 1e-9) * 2 * np.pi)  # unit: Hz
    spec_line *= 1e-6  # unit: MHz
    cav_freq = 3e8 / (WAVELENGTH * 1e-9)  # unit: Hz
    cav_freq *= 1e-6  # unit: MHz
    print("Sigma (single): {} ns".format(sigma_single))
    print("Sigma (frequency): {} MHz".format(spec_line))
    print("Q: {}".format(cav_freq / spec_line))
