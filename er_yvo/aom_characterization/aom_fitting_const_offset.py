import glob
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, ConstantModel


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab"
            "/Lab data/Er YVO Holeburning/aom/cascaded_AOMs/112ns_offset")
OUTPUT_DIR = "/output_figs/aom_fitting/112ns_offset"
FILES_IGNORE = ["30ns/NewFile1.csv",
                "30ns/NewFile2.csv"]
GAIN = 700  # units: V/W

# plotting parameters
mpl.rcParams.update({'font.size': 12,
                     'figure.figsize': (8, 6)})


# get all dataframes
filenames = glob.glob('**/*.csv', root_dir=DATA_DIR, recursive=True)
for file in FILES_IGNORE:
    filenames.remove(file)
desired_pulse_width = [os.path.split(filename)[0] for filename in filenames]
paths = [os.path.join(DATA_DIR, filename) for filename in filenames]

dfs = [pd.read_csv(path, skiprows=[1]) for path in paths]

# determine timing
time_arrays = []
for i, path in enumerate(paths):
    df_temp = pd.read_csv(path)
    increment = df_temp["Increment"][0] * 1e9  # convert to ns
    # increments.append(increment)
    num_points = len(dfs[i]["CH1"])
    time_array = np.linspace(0, (num_points-1)*increment, num_points)
    time_arrays.append(time_array)

# convert optical units
for df in dfs:
    df["CH3"] = df["CH3"].apply(lambda x: 1e3*(x/GAIN))  # convert to mW


# do fitting
hdawg_fits = []
aom_fits = []
model = GaussianModel() + ConstantModel()
for i, df in enumerate(dfs):
    time_array = time_arrays[i]
    center_guess = time_array[np.argmax(df["CH1"])]
    hdawg_fit = model.fit(df["CH1"], x=time_array,
                          c=-1, center=center_guess)
    hdawg_fits.append(hdawg_fit)

    center_guess = time_array[np.argmax(df["CH3"])]
    aom_fit = model.fit(df["CH3"], x=time_array,
                        c=0, center=center_guess)
    aom_fits.append(aom_fit)


# plot individual widths
for i, df in enumerate(dfs):
    num_points = len(df["CH1"])
    time_array = time_arrays[i]

    fig, ax = plt.subplots()
    ax1 = ax.twinx()

    ax.plot(time_array, df["CH1"], label="HDAWG Output", color='tab:blue')
    ax1.plot(time_array, df["CH3"], label="AOM Output", color='tab:orange')
    ax.plot(time_array, hdawg_fits[i].best_fit,
            'k--', label="HDAWG Fit")
    ax1.plot(time_array, aom_fits[i].best_fit,
             'k:', label="AOM Fit")

    hdawg_lw = hdawg_fits[i].params['fwhm'].value
    hdawg_err = hdawg_fits[i].params['fwhm'].stderr
    aom_lw = aom_fits[i].params['fwhm'].value
    aom_err = aom_fits[i].params['fwhm'].stderr
    res_str = rf"HDAWG FWHM: {hdawg_lw:0.3} $\pm$ {hdawg_err:0.3} ns"
    res_str += "\n"
    res_str += rf"AOM FWHM: {aom_lw:0.3} $\pm$ {aom_err:0.3} ns"
    ax.text(200, 0, res_str)

    ax.set_title(f"{desired_pulse_width[i]} pulse width")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("HDAWG Voltage (V)")
    ax1.set_ylabel("Optical Output Power (mW)")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    output_filename = os.path.join(OUTPUT_DIR, f"{desired_pulse_width[i]}.png")
    plt.savefig(output_filename)
    plt.clf()

# plot of size versus width
def get_area(x):
    return x.params['amplitude'].value

def get_area_err(x):
    return x.params['amplitude'].stderr

def get_height(x):
    return x.params['height'].value

def get_height_err(x):
    return x.params['height'].stderr

def get_lw(x):
    return x.params['fwhm'].value

def get_lw_err(x):
    return x.params['fwhm'].stderr


widths = list(map(get_lw, aom_fits))
# width_err = list(map(get_lw_err, aom_fits))
heights = list(map(get_height, aom_fits))
# height_err = list(map(get_height_err, aom_fits))
areas = list(map(get_area, aom_fits))


# plot both on same graph
fig, ax = plt.subplots()
ax1 = ax.twinx()

color_height = 'tab:blue'
color_area = 'tab:orange'

# ax.errorbar(widths, heights, xerr=width_err, yerr=height_err,
#             ls='', marker='')
ax.loglog(widths, heights, 'o', color=color_height)
ax1.loglog(widths, areas, 'o', color=color_area)

ax.set_xlabel("Optical FWHM (ns)")
ax.set_ylabel("Height (mW)", color=color_height)
ax1.set_ylabel("Area (mW * ns)", color=color_area)
ax.tick_params(axis='y', colors=color_height)
ax1.tick_params(axis='y', colors=color_area)
ax.grid(True)

fig.tight_layout()
fig.show()


# plot only heights
fig, ax = plt.subplots()

ax.plot(widths, heights, 'o', color='tab:blue')

ax.set_xlabel("Optical FWHM (ns)")
ax.set_ylabel("Height (mW)")
ax.grid(True)

fig.tight_layout()
fig.show()


# plot only areas
fig, ax = plt.subplots()

ax.plot(widths, areas, 'o', color='tab:orange')

ax.set_xlabel("Optical FWHM (ns)")
ax.set_ylabel("Area (mW * ns)")
ax.grid(True)

fig.tight_layout()
fig.show()
