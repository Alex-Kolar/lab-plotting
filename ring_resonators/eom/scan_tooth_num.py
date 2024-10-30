import glob
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/eom_data/08082024/tooth_scan")

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})


# find and read spectrum analyzer files
filenames = glob.glob('*.csv', root_dir=DATA_DIR)
tooth_nums = []
data_dfs = []
for file in filenames:
    file_str = os.path.splitext(file)[0]
    tooth_nums.append(int(file_str))
    file_path = os.path.join(DATA_DIR, file)
    data_dfs.append(pd.read_csv(file_path, header=1))

# sort
tooth_nums, data_dfs = zip(*sorted(zip(tooth_nums, data_dfs)))


# plotting of each scan
for tooth_num, data_df in zip(tooth_nums, data_dfs):
    plt.plot(data_df['Freq'] / 1e6, data_df['Amp'])

    plt.title(f"N = {tooth_num}")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power (dBm)")

    plt.tight_layout()
    plt.show()


# plotting of all histograms
num_hist = len(tooth_nums)
bins = np.linspace(-87, -60, 45)
fig, axs = plt.subplots(num_hist, 1, sharex=True,
                        figsize=(8, 8))
fig.subplots_adjust(hspace=0)

for i, ax in enumerate(axs):
    ax.hist(data_dfs[i]['Amp'], bins=bins)
    ax.set_yticks([])
    label = f"N = {tooth_nums[i]}"
    t = ax.text(0.95, 0.90, label,
                horizontalalignment='right', verticalalignment='top')
    t.set_transform(ax.transAxes)

axs[0].set_title('Power Distribution')
axs[-1].set_xlabel('Power (dBm)')

fig.show()

# # plotting of each scan
# for tooth_num, data_df in zip(tooth_num, data_dfs):
#     plt.plot(data_df['Freq'] / 1e6, data_df['Amp'])
#
#     plt.title(f"{tooth_num} teeth")
#     plt.xlabel("Frequency (MHz)")
#     plt.ylabel("Power (dBm)")
#
#     plt.tight_layout()
#     plt.show()
#
#
#     plt.hist(data_df['Amp'], bins=50)
#     plt.title(f"{tooth_num} teeth")
#     plt.xlabel("Power (dBm)")
#
#     plt.tight_layout()
#     plt.show()
