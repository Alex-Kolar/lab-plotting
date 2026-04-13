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
signal_idler_offset_guess = 453
idler_lim_input = (400, 1200)


def sort_func(filename):
    file_parts = filename.split('_')
    try:
        return int(file_parts[-1].split('.')[0])
    except ValueError:
        return 0

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

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(all_hist, cmap=cmap)
ax.set_title('Summed Histogram')
ax.set_xlabel('Idler Bin')
ax.set_ylabel('Signal Bin')
fig.colorbar(im, label='Counts')
fig.tight_layout()
fig.show()

# plot summed histogram (zoom on input)
signal_start = (idler_lim_input[0] + signal_idler_offset_guess)
signal_end = (idler_lim_input[1] + signal_idler_offset_guess)
idler_start = idler_lim_input[0]
idler_end = idler_lim_input[1]
input_zoom = all_hist[signal_start:signal_end,
                      idler_start:idler_end]

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(input_zoom, cmap=cmap)
ax.set_title('Summed Histogram Input Detail')
ax.set_xlabel('Idler Bin')
ax.set_ylabel('Signal Bin')
fig.colorbar(im, label='Counts')
fig.tight_layout()
fig.show()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(np.log10(input_zoom+1), cmap=cmap)
ax.set_title('Summed Histogram Input Detail (Log Scale)')
ax.set_xlabel('Idler Bin')
ax.set_ylabel('Signal Bin')
fig.colorbar(im, label=r'$\mathrm{log}_{10}(\mathrm{counts} + 1)$')
fig.tight_layout()
fig.show()

# plot summed histogram (zoom on echo)
signal_start = (idler_lim_input[0] + signal_idler_offset_guess) + 2000
signal_end = (idler_lim_input[1] + signal_idler_offset_guess) + 2000
idler_start = idler_lim_input[0]
idler_end = idler_lim_input[1]
echo_zoom = all_hist[signal_start:signal_end,
                     idler_start:idler_end]

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(echo_zoom, cmap=cmap)
ax.set_title('Summed Histogram Echo Detail')
ax.set_xlabel('Idler Bin')
ax.set_ylabel('Signal Bin')
fig.colorbar(im, label='Counts')
fig.tight_layout()
fig.show()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(np.log10(echo_zoom+1), cmap=cmap)
ax.set_title('Summed Histogram Echo Detail (Log Scale)')
ax.set_xlabel('Idler Bin')
ax.set_ylabel('Signal Bin')
fig.colorbar(im, label=r'$\mathrm{log}_{10}(\mathrm{counts} + 1)$')
fig.tight_layout()
fig.show()


# compute diagonals
start_diag = 0
end_diag = all_hist.shape[0] - idler_lim_input[1]
offsets = np.arange(start_diag, end_diag, 1)
offsets = offsets * -1
diags = [np.diagonal(all_hist, offset=offset)[idler_lim_input[0]:idler_lim_input[1]]
         for offset in offsets]
diag_sum = np.sum(diags, axis=1)

# convert from offset to signal-idler arrival time in ns
# bin size for experiment is 500 ps
time_arr = offsets * -0.5

# plot diagonals and zoom on certain regions
plt.plot(time_arr, diag_sum)
plt.xlabel('Signal - Idler Coincidence Delay (ns)')
plt.ylabel('Counts')
plt.tight_layout()
plt.show()


input_idx = np.where(np.logical_and(time_arr > (signal_idler_offset_guess*0.5)-20,
                                    time_arr < (signal_idler_offset_guess*0.5)+20))[0]
echo_idx = input_idx + 2000

fig, ax = plt.subplots()
ax_r = ax.twinx()
ax.plot(time_arr[input_idx], diag_sum[input_idx],
        color='cornflowerblue')
ax_r.plot(time_arr[input_idx], diag_sum[echo_idx],
          color='coral')
ax.set_ylabel('Input Counts')
ax_r.set_ylabel('Echo Counts')
ax.set_xlabel('Signal - Idler Coincidence Delay (ns)')
fig.tight_layout()
fig.show()
