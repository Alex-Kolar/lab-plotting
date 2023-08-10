import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, ConstantModel


FILENAME = "/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators" \
           "/Coincidence Count Measurement/08022023/Correlation-2_2023-08-03_14-43-39_(30sec_int).txt"
FITTING = True  # add a gaussian fit

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
xlim = (-7, 7)
color = 'coral'

# NOTE: first line of csv needs extra tab added
# otherwise the columns will not be read properly
df = pd.read_csv(FILENAME, sep='\t')
coincidence = df["Counts"]
time = df["Time(ps)"]  # unit: ps
time *= 1e-3  # unit: ns
time_diff = time[1] - time[0]  # spacing of histogram

# fitting
if FITTING:
    model = GaussianModel() + ConstantModel()
    center_guess = time[coincidence.idxmax()]
    amplitude_guess = coincidence.max()
    res = model.fit(df["Counts"], x=time,
                    center=center_guess, amplitude=amplitude_guess)
    print(res.fit_report())

    sigma = res.params['sigma'].value
    sigma_single = sigma/np.sqrt(2)


# plotting
fig, ax = plt.subplots()

ax.bar(time, coincidence, width=time_diff, color=color)
if FITTING:
    ax.plot(time, res.best_fit, 'k--')
    label = r"$\sigma$: {:0.3f} $\pm$ {:0.3f} ns".format(
        res.params['sigma'].value, res.params['sigma'].stderr)
    label += "\n"
    label += r"$\sigma$ (single): {:0.3f} ns".format(
        sigma_single)
    t = ax.text(0.05, 0.95, label,
                horizontalalignment='left', verticalalignment='top')
    t.set_transform(ax.transAxes)

ax.set_xlim(xlim)
ax.set_xlabel("Timing Offset (ns)")
ax.set_ylabel("Coincidence Counts")

fig.tight_layout()
fig.show()
