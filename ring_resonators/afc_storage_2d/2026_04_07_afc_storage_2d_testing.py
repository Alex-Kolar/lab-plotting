import glob
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


DATA_DIR = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
            '/Mounted_device_mk_5/10mK/2026_04_07/afc_longterm/2D')

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
PLOT_ALL_FILES = False
files_to_include = 1

signal_idler_offset = 226


def sort_func(filename):
    file_parts = filename.split('_')
    try:
        return int(file_parts[-1].split('.')[0])
    except ValueError:
        return 0


def shrink_array(arr, factor):
    """Re-bin an array by an integer factor.
    In the new array, each bin is the sum of the original bins.
    """
    old_shape = arr.shape
    new_shape = (old_shape[0] // factor, old_shape[1] // factor)
    new_arr = np.zeros(new_shape)
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            new_arr[i, j] = np.sum(arr[i*factor:(i+1)*factor, j*factor:(j+1)*factor])
    return new_arr


all_files = glob.glob(os.path.join(DATA_DIR, '*.npz'))
all_files.sort(key=sort_func)

# load data and plot individual files
all_counts = []
all_bins = []
all_freqs = []
file_numbers = []
for file in all_files:
    try:
        file_number = int(file.split('/')[-1].split('.')[0].split('_')[-1])
    except ValueError:
        file_number = 0
    file_numbers.append(file_number)

    data = np.load(file)
    counts = data['counts']
    bins = data['bins']
    freqs_start = data['freq_start']
    freqs_end = data['freq_end']
    all_counts.append(counts)
    all_bins.append(bins)
    all_freqs.append((freqs_start, freqs_end))

# plot total number of counts from each file
all_sum = np.sum(all_counts, axis=(1,2))
plt.plot(file_numbers, all_sum,
         marker='o', color='cornflowerblue')
plt.xlabel('File Number')
plt.ylabel('Total Counts')
plt.title('Total Counts from Each File')
plt.tight_layout()
plt.show()

first_2000_sum = np.sum(all_counts[0][:2000,:2000])
print(first_2000_sum)
all_sum = np.sum(all_counts[0])
print(all_sum)

# plot frequencies
start_freqs = [freq[0] for freq in all_freqs]
end_freqs = [freq[1] for freq in all_freqs]
plt.plot(file_numbers, start_freqs,
         color='cornflowerblue', marker='o', label='Start Frequency')
plt.plot(file_numbers, end_freqs,
         color='coral', marker='o', label='End Frequency')
plt.xlabel('File Number')
plt.ylabel('Frequency (GHz)')
plt.title('Start and End Frequencies')
plt.ylim(194991.935 - 0.1, 194991.935 + 0.1)
plt.legend()
plt.tight_layout()
plt.show()

# plot summed histogram
all_hist = np.sum(all_counts[:files_to_include], axis=0)
# X, Y = np.meshgrid(all_bins[0], all_bins[1], sparse=True)
# plt.pcolormesh(X, Y, all_hist)
plt.imshow(all_hist, cmap='magma')
plt.xlim(0, 2000)
plt.ylim(0, 2000)
plt.title('Summed Histogram')
plt.xlabel('Idler Bin')
plt.ylabel('Signal Bin')
plt.colorbar()
plt.tight_layout()
plt.show()

# plot summed histogram (zoom on input)
plt.imshow(all_hist, cmap='magma')
plt.xlim(250, 550)
plt.ylim(250+signal_idler_offset, 550+signal_idler_offset)
plt.title('Summed Histogram Detail (Input)')
plt.xlabel('Idler Bin')
plt.ylabel('Signal Bin')
plt.colorbar()
plt.tight_layout()
plt.show()


# plot summed histogram (zoom on echo)
plt.imshow(all_hist, cmap='magma')
plt.xlim(250, 550)
plt.ylim(250+signal_idler_offset+1000, 550+signal_idler_offset+1000)
plt.title('Summed Histogram Detail (Echo)')
plt.xlabel('Idler Bin')
plt.ylabel('Signal Bin')
plt.colorbar()
plt.tight_layout()
plt.show()

# plot summed histogram diagonal
diag = np.diagonal(all_hist, offset=-signal_idler_offset)
plt.plot(diag)
plt.tight_layout()
plt.show()
print('diagonal sum:', np.sum(diag))

# plot log of summed histogram
epsilon = 1e-10
log_plot = np.log10(all_hist + epsilon)
plt.imshow(log_plot)
plt.xlim(0, 2000)
plt.ylim(0, 2000)
plt.title('Summed Histogram (Log Scale)')
plt.xlabel('Idler Bin')
plt.ylabel('Signal Bin')
plt.tight_layout()
plt.show()

# # re-bin and plot
# new_hist = shrink_array(all_hist, 2)
# plt.imshow(new_hist, cmap='magma')
# plt.title('Summed Histogram (1 ns bins)')
# plt.xlabel('Idler Bin')
# plt.ylabel('Signal Bin')
# plt.colorbar()
# plt.tight_layout()
# plt.show()
