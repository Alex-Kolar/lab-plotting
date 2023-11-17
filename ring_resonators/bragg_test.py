import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


DATA_PATH = "/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators/Bragg Test/"

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
no_filter_color = 'cornflowerblue'
filter_color = 'coral'


# get dark count info
df_dark = pd.read_csv(DATA_PATH + "dark.txt", sep='\t')
dark_avg = np.mean(df_dark['counts'])
dark_std = np.std(df_dark['counts'])

# get info for off-resonant
df_off = pd.read_csv(DATA_PATH + "off_res_no_filter.txt", sep='\t')
off_avg = np.mean(df_off['counts'])
off_std = np.std(df_off['counts'])
df_off_filter = pd.read_csv(DATA_PATH + "off_res_filter.txt", sep='\t')
off_filter_avg = np.mean(df_off_filter['counts'])
off_filter_std = np.std(df_off_filter['counts'])

# get info for filter center
df_center = pd.read_csv(DATA_PATH + "filter_center_no_filter.txt", sep='\t')
center_avg = np.mean(df_center['counts'])
center_std = np.std(df_center['counts'])
df_center_filter = pd.read_csv(DATA_PATH + "filter_center_filter.txt", sep='\t')
center_filter_avg = np.mean(df_center_filter['counts'])
center_filter_std = np.std(df_center_filter['counts'])

# get info for on-resonant
df_on = pd.read_csv(DATA_PATH + "on_res_no_filter.txt", sep='\t')
on_avg = np.mean(df_on['counts'])
on_std = np.std(df_on['counts'])
df_on_filter = pd.read_csv(DATA_PATH + "on_res_filter.txt", sep='\t')
on_filter_avg = np.mean(df_on_filter['counts'])
on_filter_std = np.std(df_on_filter['counts'])


# plotting
x = np.arange(3)
width = 0.4
x_no_filter = x - width / 2
x_filter = x + width / 2

# plotting of no filter
plt.bar(x_no_filter, [off_avg, center_avg, on_avg],
        width=width, edgecolor='k', color=no_filter_color, zorder=2,
        label="No Bragg Grating")
plt.errorbar(x_no_filter, [off_avg, center_avg, on_avg],
             yerr=[off_std, center_std, on_std],
             color='k', ls='', capsize=5)

# plotting of filter
plt.bar(x_filter, [off_filter_avg, center_filter_avg, on_filter_avg],
        width=width, edgecolor='k', color=filter_color, zorder=2,
        label="With Bragg Grating")
plt.errorbar(x_filter, [off_filter_avg, center_filter_avg, on_filter_avg],
             yerr=[off_filter_std, center_filter_std, on_filter_std],
             color='k', ls='', capsize=5)

# # dark counts
# plt.axhline(dark_avg, label="Dark Counts")

plt.title("Bragg filter testing")
x_label = ["1540.32 nm Pump", "1541.32 nm Pump", "1541.772 nm Pump"]
plt.xticks(x, x_label)
plt.ylabel("Counts/s")
plt.legend(shadow=True)
plt.grid(axis='y')

plt.tight_layout()
plt.show()
