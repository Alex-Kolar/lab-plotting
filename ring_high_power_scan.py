import glob
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = "/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators/07072023_power dependent"
OUTPUT_FILENAME = "output_figs/ring_high_power_scan.png"

# data taken for frequency
F_MIN = 193395.328
F_MAX = 193426.015

# plotting parameters
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
x_range = (15.7, 19)  # unit: GHz
y_range = (0, 0.175)


# locate all files
csv_files = glob.glob('*mw_high.csv', root_dir=DATA_DIR)
csv_paths = [os.path.join(DATA_DIR, file) for file in csv_files]

# sort all files and get powers
powers = np.zeros(len(csv_files))
for i, file in enumerate(csv_files):
    power_str = file[:file.find("mw")]
    power = float(power_str)
    powers[i] = power
csv_paths = sorted(csv_paths)
powers = sorted(powers)
print(sorted(csv_files))

dfs = [pd.read_csv(path, header=11) for path in csv_paths]

# get frequency scan info
df = dfs[0]
scan = df['Volt']
id_min = scan.idxmin()
id_max = scan.idxmax()


# plotting
fig, ax = plt.subplots()

cmap = mpl.cm.get_cmap('Reds')
for power, df in zip(powers, dfs):
    scan = df['Volt']
    id_min = scan.idxmin()
    id_max = scan.idxmax()

    # convert to frequency
    freq_start = 0
    freq_stop = F_MAX - F_MIN
    freq = np.linspace(freq_start, freq_stop, num=(id_max-id_min))

    column = df['Volt.1'][id_min:id_max]
    column = column.reset_index(drop=True)
    min_trans = column.min()
    min_freq = freq[column.idxmin()]

    color = cmap(power / max(powers))
    ax.plot(freq, column,
            color=color)
    # ax.axvline(x=min_freq, color='k')
    # ax.text(min_freq-0.05, min_trans-0.015, r'{:0.2f} mW'.format(power),
    #         bbox={'facecolor': 'white', 'pad': 3}, fontsize=10)

# ax.set_xlim(x_range)
# ax.set_ylim(y_range)
ax.set_title("Cavity Resonance Shift (High Power)")
ax.set_xlabel("Detuning from {:0.0f} GHz (GHz)".format(F_MIN))
ax.set_ylabel("Transmission (A.U.)")

fig.tight_layout()
plt.savefig(OUTPUT_FILENAME)
