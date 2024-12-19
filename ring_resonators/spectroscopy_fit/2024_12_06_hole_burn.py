"""Data from December 6 2024

Sequence of data files:
1. PREBURN: Hyperfine polarized spectrum
2. PREBURNZOOM: Same as above, but with smaller scan range for higher resolution
3. AFTERBURN: Same range as PREBURNZOOM, but after using unlocked laser to burn hole
4. AFTERHOLE: Approx. same range as AFTERBURN, but after re-polarizing the hyperfine spectrum
5. AFTERBURNLOCK: Approx. same range, but after using locked laser to burn hole
"""

import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# data files
BG = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
      "/Bulk_crystal/10mK/12032024/LASER_OFF.csv")
DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data"
            "/Ring Resonators/Bulk_crystal/10mK/12062024")
SEQUENCE = ["PREBURN",
            "PREBURNZOOM",
            "AFTERBURN",
            "AFTERHOLE",
            "AFTERBURNLOCK"]
TITLES = ["Before Burning",
          "Before Burning (Zoom)",
          "After Burning (Unlocked)",
          "After Re-initialization",
          "After Burning (Locked)"]
# scan ranges used to take data (GHz)
RANGES = [(194808.510, 194817.177),
          (194812.442, 194813.162),
          (194812.442, 194813.162),
          (194812.478, 194813.183),
          (194812.656, 194812.999)]

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
color_od = 'coral'
SAVE_FILE = True
OUTPUT_DIR = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
              "/bulk_crystal/10mK/hole_burning/2024_12_06_piezo")


# reference data
df_laser_off = pd.read_csv(BG, header=10, skiprows=[11])
off_level = np.mean(df_laser_off['CH2'].astype(float).to_numpy())
df_ref = pd.read_csv(os.path.join(DATA_DIR, SEQUENCE[0] + '.csv'), header=10, skiprows=[11])
transmission_ref = df_ref['CH2'].astype(float).to_numpy()
transmission_ref -= off_level
bg = max(transmission_ref)

# experiment data
dfs = []
freqs = []
trans = []
ods = []
for filename, scan_range in zip(SEQUENCE, RANGES):
    df = pd.read_csv(os.path.join(DATA_DIR, filename + '.csv'), header=10, skiprows=[11])

    ramp = df['CH1'].astype(float).to_numpy()
    transmission = df['CH2'].astype(float).to_numpy()
    transmission -= off_level

    id_min = np.argmin(ramp)
    id_max = np.argmax(ramp)
    transmission = transmission[id_min:id_max]
    trans.append(transmission)

    # convert time to frequency
    freq = np.linspace(0, (scan_range[1] - scan_range[0]), id_max - id_min)  # unit: GHz
    freqs.append(freq)

    # convert to optical depth
    od = np.log(bg / transmission)
    ods.append(od)

# plot data
for i in range(len(SEQUENCE)):
    # plot transmission
    plt.plot(freqs[i], trans[i],
             color=color, label="Transmission")
    plt.title(TITLES[i])
    plt.xlabel(f"Frequency + {RANGES[i][0]:.3f} (GHz)")
    plt.ylabel("Transmission (A.U.)")

    plt.tight_layout()
    if SAVE_FILE:
        savename = f"2024_12_06_{SEQUENCE[i]}.png"
        savepath = os.path.join(OUTPUT_DIR, savename)
        plt.savefig(savepath)
    else:
        plt.show()

    # plot od
    plt.plot(freqs[i], ods[i],
             color=color_od, label="OD")
    plt.title(TITLES[i])
    plt.xlabel(f"Frequency + {RANGES[i][0]:.3f} (GHz)")
    plt.ylabel("Optical Depth")
    plt.ylim((0, 4))

    plt.tight_layout()
    if SAVE_FILE:
        savename = f"2024_12_06_{SEQUENCE[i]}_OD.png"
        savepath = os.path.join(OUTPUT_DIR, savename)
        plt.savefig(savepath)
    else:
        plt.show()
