import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, ConstantModel


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab"
            "/Lab data/Er YVO Holeburning/aom/cascaded_AOMs/112ns_offset")
OUTPUT_DIR = "/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/aom_fitting/112ns_offset"
FILES_IGNORE = ["30ns/NewFile1.csv",
                "30ns/NewFile2.csv"]


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


for i, df in enumerate(dfs):
    num_points = len(df["CH1"])
    time_array = time_arrays[i]

    plt.plot(time_array, df["CH1"], label="HDAWG Output")
    plt.plot(time_array, df["CH3"], label="AOM Output")
    plt.plot(time_array, hdawg_fits[i].best_fit,
             'k--', label="HDAWG Fit")
    plt.plot(time_array, aom_fits[i].best_fit,
             'k:', label="AOM Fit")

    hdawg_lw = hdawg_fits[i].params['fwhm'].value
    hdawg_err = hdawg_fits[i].params['fwhm'].stderr
    aom_lw = aom_fits[i].params['fwhm'].value
    aom_err = aom_fits[i].params['fwhm'].stderr
    res_str = rf"HDAWG FWHM: {hdawg_lw:0.3} $\pm$ {hdawg_err:0.3} ns"
    res_str += "\n"
    res_str += rf"AOM FWHM: {aom_lw:0.3} $\pm$ {aom_err:0.3} ns"
    plt.text(300, -0.75, res_str)

    plt.title(f"{desired_pulse_width[i]} pulse width")
    plt.xlabel("Time (ns)")

    plt.grid(True)
    plt.legend()

    output_filename = os.path.join(OUTPUT_DIR, f"{desired_pulse_width[i]}.png")
    plt.savefig(output_filename)
    plt.clf()
