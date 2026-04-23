import glob
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors


DATA_DIR = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
            '/Mounted_device_mk_5/10mK/2026_04_07/afc_longterm/2D')

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
files_to_include = 16
cmap_name = 'magma'
log_plot = True

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap(cmap_name)
new_cmap = truncate_colormap(cmap, minval=0.1, maxval=1.0)

# data processing params
signal_idler_offset = 463  # all for 500 ps bins
storage_time = 2000
# idler_lim_input = (400, 1200)
idler_lim_input = (300, 1300)
scaling_factor = 20  # number of (500 ps) bins to include
bin_width = 0.5


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

# plot summed histogram
all_hist = np.sum(all_counts[:files_to_include], axis=0)
if scaling_factor > 1:
    all_hist = shrink_array(all_hist, scaling_factor)

if log_plot:
    plt.imshow(all_hist, cmap=cmap,
               norm=colors.LogNorm(vmin=1, vmax=all_hist.max()))
else:
    plt.imshow(all_hist, cmap=cmap)
plt.title('Summed Histogram')
plt.xlabel('Idler Bin')
plt.ylabel('Signal Bin')
plt.colorbar(label='Counts')
plt.tight_layout()
plt.show()

# plot summed histogram (zoom on input, log scale)
signal_start = (idler_lim_input[0] + signal_idler_offset) // scaling_factor
signal_end = (idler_lim_input[1] + signal_idler_offset) // scaling_factor
idler_start = idler_lim_input[0] // scaling_factor
idler_end = idler_lim_input[1] // scaling_factor
input_zoom = all_hist[signal_start:signal_end,
                      idler_start:idler_end]

idler_time_values = np.arange(idler_start, idler_end) * scaling_factor * bin_width
signal_time_values = np.arange(signal_start, signal_end) * scaling_factor * bin_width
X, Y = np.meshgrid(idler_time_values, signal_time_values)

fig, ax = plt.subplots(figsize=(5, 4), dpi=400)
if log_plot:
    plt.pcolormesh(X, Y, input_zoom, cmap=cmap,
                   norm=colors.LogNorm(vmin=1, vmax=all_hist.max()))
else:
    plt.pcolormesh(X, Y, input_zoom, cmap=cmap)
ax.axes.set_aspect('equal')
plt.title('AFC Reflected Input')
plt.xlabel('Idler Time (ns)')
plt.ylabel('Signal Time (ns)')
plt.colorbar(label='Counts')
plt.tight_layout()
plt.show()

# plot summed histogram (zoom on echo)
signal_start = (idler_lim_input[0] + signal_idler_offset + storage_time) // scaling_factor
signal_end = (idler_lim_input[1] + signal_idler_offset + storage_time) // scaling_factor
echo_zoom = all_hist[signal_start:signal_end,
                     idler_start:idler_end]

Y_echo = Y + (storage_time * bin_width)

fig, ax = plt.subplots(figsize=(5, 4), dpi=400)
if log_plot:
    plt.pcolormesh(X, Y_echo, echo_zoom, cmap=cmap,
                   norm=colors.LogNorm(vmin=1, vmax=all_hist.max()))
else:
    plt.pcolormesh(X, Y_echo, echo_zoom, cmap=cmap)
ax.axes.set_aspect('equal')
plt.title('AFC Echo')
plt.xlabel('Idler Time (ns)')
plt.ylabel('Signal Time (ns)')
plt.colorbar(label='Counts')
plt.tight_layout()
plt.show()


# plot both histograms side-by-side
# transpose signal and idler to better fit in plot
X = np.swapaxes(X, 0, 1)
Y = np.swapaxes(Y, 0, 1)
Y_echo = np.swapaxes(Y_echo, 0, 1)
fig, axs = plt.subplots(1, 2, figsize=(9, 4), dpi=400)
if log_plot:
    axs[0].pcolormesh(Y, X, input_zoom.transpose(), cmap=new_cmap,
                      norm=colors.LogNorm(vmin=1, vmax=all_hist.max()))
else:
    axs[0].pcolormesh(Y, X, input_zoom.transpose(), cmap=cmap)
axs[0].set_title('AFC Reflected Input')
axs[0].set_xlabel('Signal Time (ns)')
axs[0].set_ylabel('Idler Time (ns)')
axs[0].axes.set_aspect('equal')
axs[0].set_facecolor('black')
if log_plot:
    axs[1].pcolormesh(Y_echo, X, echo_zoom.transpose(), cmap=new_cmap,
                      norm=colors.LogNorm(vmin=1, vmax=all_hist.max()))
else:
    axs[1].pcolormesh(Y_echo, X, echo_zoom.transpose(), cmap=cmap)
axs[1].set_title('AFC Echo')
axs[1].set_xlabel('Signal Time (ns)')
axs[1].axes.set_aspect('equal')
axs[1].set_facecolor('black')
axs[1].set_yticks([])
fig.colorbar(axs[0].get_children()[0], ax=axs[1], label='Counts')

fig.tight_layout()
fig.show()
