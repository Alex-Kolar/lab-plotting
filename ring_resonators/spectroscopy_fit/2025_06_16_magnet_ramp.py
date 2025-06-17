import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/10mK/06162025")
DATA_FILES = ["SDS00001.csv",
              "SDS00002.csv",
              "SDS00004.csv"]
FREQ_RANGE = [(194818.709, 194827.320),
              (194824.728, 194833.275),
              (194804.342, 194812.853)]  # unit: GHz
AOM_OFFSET = 0.600  # unit: GHz

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
titles = [r'Spectrum at 900 mT along $(1, 0, 0)_{D_1, D_2, b}$',
          r'Spectrum at 1 T along $(-1, 1, 0)_{D_1, D_2, b}$',
          r'Spectrum at 0 T']
ref_freq = 194800  # unit: GHz
color = 'cornflowerblue'


# read data files
dfs = []
freqs = []
transmissions = []
for file, freq_range in zip(DATA_FILES, FREQ_RANGE):
    path = os.path.join(DATA_DIR, file)
    df = pd.read_csv(path, header=10, skiprows=[11])
    dfs.append(df)

    # get ramp data from first file
    ramp = df['CH1'].astype(float).to_numpy()
    id_min = np.argmin(ramp)
    id_max = np.argmax(ramp)

    # convert time to frequency
    freq = np.linspace(freq_range[0], freq_range[1], id_max - id_min)  # unit: GHz
    freq += AOM_OFFSET
    freq -= ref_freq
    freqs.append(freq)

    transmission = df['CH2']
    transmission = transmission[id_min:id_max]
    transmissions.append(transmission)


# plot all (for reference)
for freq, transmission, title in zip(freqs, transmissions, titles):
    fig, ax = plt.subplots()
    ax.plot(freq, transmission)
    ax.set_title(title)
    ax.set_xlabel(f'Frequency - {ref_freq} (GHz)')
    ax.set_ylabel(f'Transmission (A.U.)')
    fig.tight_layout()
    fig.show()
