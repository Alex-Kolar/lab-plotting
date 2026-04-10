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
files_to_include = 16
cmap = 'magma'

# data processing params
signal_idler_offset = 453  # all for 500 ps bins
storage_time = 2000
idler_lim_input = (300, 1400)
scaling_factor = 20  # number of (500 ps) bins to include


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

# load data
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
if scaling_factor > 1:
    all_hist = shrink_array(all_hist, scaling_factor)

plt.imshow(all_hist, cmap=cmap)
plt.title('Summed Histogram')
plt.xlabel('Idler Bin')
plt.ylabel('Signal Bin')
plt.colorbar(label='Counts')
plt.tight_layout()
plt.show()

# plot summed histogram (zoom on input)
signal_start = (idler_lim_input[0] + signal_idler_offset) // scaling_factor
signal_end = (idler_lim_input[1] + signal_idler_offset) // scaling_factor
idler_start = idler_lim_input[0] // scaling_factor
idler_end = idler_lim_input[1] // scaling_factor
input_zoom = all_hist[signal_start:signal_end,
                      idler_start:idler_end]
plt.imshow(input_zoom, cmap=cmap)
plt.title('Summed Histogram Detail (AFC Input)')
plt.xlabel('Idler Bin')
plt.ylabel('Signal Bin')
plt.colorbar(label='Counts')
plt.tight_layout()
plt.show()

# plot summed histogram (zoom on echo)
signal_start = (idler_lim_input[0] + signal_idler_offset + storage_time) // scaling_factor
signal_end = (idler_lim_input[1] + signal_idler_offset + storage_time) // scaling_factor
echo_zoom = all_hist[signal_start:signal_end,
                     idler_start:idler_end]
plt.imshow(echo_zoom, cmap=cmap)
plt.title('Summed Histogram Detail (AFC Echo)')
plt.xlabel('Idler Bin')
plt.ylabel('Signal Bin')
plt.colorbar(label='Counts')
plt.tight_layout()
plt.show()

# plot summed histogram diagonal for several candidate bins (input)
offsets = np.array([-(signal_idler_offset // scaling_factor) + i for i in range(-5, 6)])
start = idler_lim_input[0] // scaling_factor
end = idler_lim_input[1] // scaling_factor
diagonals = [np.diagonal(all_hist, offset=offset)[start:end] for offset in offsets]
diagonal_sums = np.sum(diagonals, axis=1)

offsets_for_plot = offsets * (scaling_factor * 0.5)  # convert to ns
plt.bar(offsets_for_plot, diagonal_sums,
        width=(0.8*scaling_factor*0.5))
plt.xlabel('Signal-Idler Offset (ns)')
plt.ylabel('Coincidences')
plt.title('Recovered Coincidence Histogram (Input)')
plt.tight_layout()
plt.show()

# collect max
max_index = np.argmax(diagonal_sums)
max_offset = offsets[max_index]
max_diag = diagonals[max_index]

# plot summed histogram diagonal (input)
time_for_diag = np.linspace(start, end, len(max_diag)) * scaling_factor * 0.5  # convert to ns
plt.plot(time_for_diag, max_diag)
plt.axhline(0, color='k')
plt.xlabel('Idler Arrival Time (ns)')
plt.ylabel('Counts')
plt.title('Input Time-Resolved Coincidences')
plt.tight_layout()
plt.show()


# plot summed histogram diagonal for several candidate bins (echo)
offsets = offsets - (storage_time // scaling_factor)
diagonals = [np.diagonal(all_hist, offset=offset)[start:end] for offset in offsets]
diagonal_sums = np.sum(diagonals, axis=1)

offsets_for_plot = np.array(offsets) * (scaling_factor * 0.5)  # convert to ns
plt.bar(offsets_for_plot, diagonal_sums,
        width=(0.8*scaling_factor*0.5))
plt.xlabel('Signal-Idler Offset (ns)')
plt.ylabel('Coincidences')
plt.title('Recovered Coincidence Histogram (Echo)')
plt.tight_layout()
plt.show()

# plot summed histogram diagonal (echo)
offset_echo = max_offset - (storage_time // scaling_factor)
max_diag_echo = np.diagonal(all_hist, offset=offset_echo)[start:end]

plt.plot(time_for_diag, max_diag_echo, label='Echo Coincidence Diagonal')
plt.axhline(0, color='k', label='0 Counts')
plt.axhline(10, color='k', ls='--', label='10 Counts')
plt.xlabel('Idler Arrival Time (ns)')
plt.ylabel('Counts')
plt.title('Echo Time-Resolved Coincidences')
plt.legend()
plt.tight_layout()
plt.show()

# plot log of summed histogram
epsilon = 1
log_plot = np.log10(all_hist + epsilon)
plt.imshow(log_plot)
plt.title('Summed Histogram (Log Scale)')
plt.xlabel('Idler Bin')
plt.ylabel('Signal Bin')
plt.colorbar(label=r'$\mathrm{log}_{10}(\mathrm{counts} + 1)$')
plt.tight_layout()
plt.show()
