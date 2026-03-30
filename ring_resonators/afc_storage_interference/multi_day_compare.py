import glob
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


DIRS_CONSTRUCTIVE = [('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
                      '/Mounted_device_mk_5/10mK/2026_03_23/afc_longterm'),
                     ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
                      '/Mounted_device_mk_5/10mK/2026_03_23/afc_longterm'),
                     ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
                      '/Mounted_device_mk_5/10mK/2026_03_27/afc_longterm')]
# FILES_TO_INCLUDE_CONSTRUCTIVE = [12, 6, 2]
FILES_TO_INCLUDE_CONSTRUCTIVE = [5, 5, 0]
DIRS_DESTRUCTIVE = [('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
                     '/Mounted_device_mk_5/10mK/2026_03_21/afc_longterm'),
                    ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
                     '/Mounted_device_mk_5/10mK/2026_03_25/afc_longterm')]
# FILES_TO_INCLUDE_DESTRUCTIVE = [12, 15]
FILES_TO_INCLUDE_DESTRUCTIVE = [5, 5]
center = 0.22599999999999998

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color_constructive = 'cornflowerblue'
color_destructive = 'coral'
xlim_input = (-0.1, 0.1)
xlim_echo = (0.9, 1.1)
ylim_input = (0, 1200)
ylim_echo = (0, 150)

# ranges for noise calculation
bg_range = (-1.1, -0.9)
window_range = (0.998, 1.005)
bg_noise_range_1 = (1.025, 1.1)
bg_noise_range_2 = (0.9, 0.975)



def sort_func(filename):
    file_parts = filename.split('_')
    try:
        return int(file_parts[-1].split('.')[0])
    except ValueError:
        return 0


all_counts_constructive = []
all_bins_constructive = []
for num_files, dir in zip(FILES_TO_INCLUDE_CONSTRUCTIVE, DIRS_CONSTRUCTIVE):
    all_files = glob.glob(os.path.join(dir, '*.npz'))
    all_files.sort(key=sort_func)

    # load data and plot individual files (constructive)
    all_bins = []
    all_counts = []
    all_freqs = []
    all_file_numbers = []
    for file in all_files:
        file_number = sort_func(file)
        all_file_numbers.append(file_number)

        data = np.load(file)
        counts = data['counts']
        bins = data['bins']
        freqs_start = data['freq_start']
        freqs_end = data['freq_end']
        all_counts.append(counts)
        all_bins.append(bins)
        all_freqs.append((freqs_start, freqs_end))

    all_counts_constructive += all_counts[:num_files]
    all_bins_constructive += all_bins[:num_files]


all_counts_destructive = []
all_bins_destructive = []
for num_files, dir in zip(FILES_TO_INCLUDE_DESTRUCTIVE, DIRS_DESTRUCTIVE):
    all_files = glob.glob(os.path.join(dir, '*.npz'))
    all_files.sort(key=sort_func)

    # load data and plot individual files (constructive)
    all_bins = []
    all_counts = []
    all_freqs = []
    all_file_numbers = []
    for file in all_files:
        file_number = sort_func(file)
        all_file_numbers.append(file_number)

        data = np.load(file)
        counts = data['counts']
        bins = data['bins']
        freqs_start = data['freq_start']
        freqs_end = data['freq_end']
        all_counts.append(counts)
        all_bins.append(bins)
        all_freqs.append((freqs_start, freqs_end))

    all_counts_destructive += all_counts[:num_files]
    all_bins_destructive += all_bins[:num_files]

# display number of files
print('Number of Constructive Files: ', len(all_counts_constructive))
print('Number of Destructive Files: ', len(all_counts_destructive))
# TODO: delete
# scaling_factor = len(all_counts_destructive) / len(all_counts_constructive)
# scaling_factor = 1.6
# print('Scaling factor for number of files:', scaling_factor)

# find center using constructive interference
bins = all_bins_constructive[0]
center = bins[np.argmax(all_counts_constructive[0])]
bins -= center

all_hist_constructive = np.sum(all_counts_constructive, axis=0)
all_hist_destructive = np.sum(all_counts_destructive, axis=0)
# TODO: delete
# all_hist_constructive = all_hist_constructive.astype(float)
# all_hist_constructive *= scaling_factor

plt.plot(bins, all_hist_constructive,
         color=color_constructive, label='Constructive')
plt.plot(bins, all_hist_destructive,
         color=color_destructive, label='Destructive')
plt.tight_layout()
plt.show()

# plot just input
plt.plot(bins, all_hist_constructive,
         color=color_constructive, label='Constructive')
plt.plot(bins, all_hist_destructive,
         color=color_destructive, label='Destructive')
plt.xlim(xlim_input)
plt.ylim(ylim_input)
plt.xlabel(r'Time ($\mathrm{\mu}$s)')
plt.ylabel('Counts')
plt.title('Input Interference')
plt.legend()
plt.tight_layout()
plt.show()

# plot just echo
plt.plot(bins, all_hist_constructive,
         color=color_constructive, label='Constructive')
plt.plot(bins, all_hist_destructive,
         color=color_destructive, label='Destructive')
plt.xlim(xlim_echo)
plt.ylim(ylim_echo)
plt.ylim(ylim_echo)
plt.xlabel(r'Time ($\mathrm{\mu}$s)')
plt.ylabel('Counts')
plt.title('Echo Interference')
plt.legend()
plt.tight_layout()
plt.show()


# calculate visibility and error
# background subtraction
bg_idx = np.where((bins >= bg_range[0]) & (bins <= bg_range[1]))[0]
echo_idx = np.where((bins >= xlim_echo[0]) & (bins <= xlim_echo[1]))[0]
bins_echo = bins[echo_idx]
echo_constructive_no_bg = all_hist_constructive[echo_idx] - all_hist_constructive[bg_idx]
echo_destructive_no_bg = all_hist_destructive[echo_idx] - all_hist_destructive[bg_idx]

# calculate visibility
window_idx = np.where((bins_echo >= window_range[0]) & (bins_echo <= window_range[1]))[0]
print('Time bins considered for visibility:', bins_echo[window_idx])
constructive_counts = np.sum(echo_constructive_no_bg[window_idx], axis=0)
destructive_counts = np.sum(echo_destructive_no_bg[window_idx], axis=0)
count_sum = constructive_counts + destructive_counts
count_diff = constructive_counts - destructive_counts
visibility = (constructive_counts - destructive_counts) / (constructive_counts + destructive_counts)
print('Constructive counts (no bg):', constructive_counts)
print('Destructive counts (no bg):', destructive_counts)
print(f'Visibility: {visibility*100:.2f}%')

# calculate noise
noise_idx_1 = np.where((bins_echo >= bg_noise_range_1[0]) & (bins_echo <= bg_noise_range_1[1]))[0]
noise_idx_2 = np.where((bins_echo >= bg_noise_range_2[0]) & (bins_echo <= bg_noise_range_2[1]))[0]
total_noise_idx = np.concatenate((noise_idx_1, noise_idx_2))
noise_std = np.std(echo_destructive_no_bg[total_noise_idx])
print(f'Noise standard deviation: {noise_std:.2f}')
destructive_std = np.sqrt(len(window_idx)) * noise_std
constructive_std = np.sqrt((len(window_idx) * (noise_std**2)) + constructive_counts)
combined_std = np.sqrt(constructive_std**2 + destructive_std**2)
# visibility_err = visibility * np.sqrt(((np.sqrt(2) * combined_std) / count_diff)**2 + ((np.sqrt(2) * combined_std) / count_sum)**2)
visibility_err = visibility * np.sqrt(((combined_std) / count_diff)**2 + ((combined_std) / count_sum)**2)
print(f'Visibility error: {visibility_err*100:.2f}%')

# plotting of background subtraction
plt.plot(bins[echo_idx], echo_constructive_no_bg,
         color=color_constructive, label='Constructive')
plt.plot(bins[echo_idx], echo_destructive_no_bg,
         color=color_destructive, label='Destructive')
plt.xlim(xlim_echo)
plt.ylim(-20, 100)
plt.xlabel(r'Time ($\mathrm{\mu}$s)')
plt.ylabel('Counts')
plt.title('Echo Interference (No Background)')
plt.legend()
plt.tight_layout()
plt.show()
