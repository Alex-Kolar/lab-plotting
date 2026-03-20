import glob
import os
import numpy as np
import matplotlib as mpl
import matplotlib.ticker as tck
import matplotlib.pyplot as plt
from lmfit.models import SineModel, ConstantModel


DATA_DIR = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
            '/Mounted_device_mk_5/10mK/2026_03_18/afc_longterm')

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
PLOT_ALL_FILES = False
xlim_input = (-0.1, 0.1)
xlim_echo = (0.9, 1.1)


all_files = glob.glob(os.path.join(DATA_DIR, '*.npz'))
all_files.sort()

# load data and plot individual files
all_counts = []
all_bins = []
file_numbers = []
for file in all_files:
    file_number = int(file.split('/')[-1].split('.')[0].split('_')[-1])
    file_numbers.append(file_number)

    data = np.load(file)
    counts = data['counts']
    bins = data['bins']
    all_counts.append(counts)
    all_bins.append(bins)

    if PLOT_ALL_FILES:
        plt.plot(bins, counts,
                 color='cornflowerblue')
        plt.title(f'AFC Experiment {file_number}')
        plt.show()

# get data from first file
bins = all_bins[0]
center = bins[np.argmax(all_counts[0])]
bins -= center

# plot total number of counts from each files
all_sum = np.sum(all_counts, axis=1)
plt.plot(file_numbers, all_sum,
         marker='o', color='cornflowerblue')
# plt.ylim(0, 24000)
plt.xlabel('File Number')
plt.ylabel('Total Counts')
plt.title('Total Counts from Each File')
plt.tight_layout()
plt.show()

# plot summed histogram
all_hist = np.sum(all_counts[:4], axis=0)
plt.plot(bins, all_hist,
         color='coral')
plt.xlabel(r'Time ($\mathrm{\mu}$s)')
plt.ylabel('Counts')
plt.title('Summed Histogram')
plt.tight_layout()
plt.show()

plt.plot(bins, all_hist,
         color='coral')
plt.xlim(xlim_input)
plt.ylim(0, 500)
plt.xlabel(r'Time ($\mathrm{\mu}$s)')
plt.ylabel('Counts')
plt.title('Input Interference')
plt.tight_layout()
plt.show()

plt.plot(all_bins[0], all_hist,
         color='coral')
plt.xlim(xlim_echo)
plt.ylim(0, 100)
plt.xlabel(r'Time ($\mathrm{\mu}$s)')
plt.ylabel('Counts')
plt.title('Echo Interference')
plt.tight_layout()
plt.show()
