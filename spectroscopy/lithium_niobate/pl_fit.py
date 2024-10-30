import glob
import numpy as np
from lmfit.models import ConstantModel, ExponentialModel
import matplotlib as mpl
import matplotlib.pyplot as plt


# data params
DATA_PATH = "/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Lithium Niobate/PL/Chip 4"

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'coral'


def draw_hist_short_long(folder_path, keyword="", factor=1, floor_counts_per_bin=0):
    files = glob.glob(folder_path + "/*.npz")
    num_files = len(files)

    PL_counts = []
    PL_shorts = []
    PL_longs = []

    laser_wls = []
    actual_freqs = []
    powers = []
    widths = []

    T1_exps_short = []
    T1_lins_short = []
    T1_exps_long = []
    T1_lins_long = []

    sd_exps_short = []
    sd_lins_short = []
    sd_exps_long = []
    sd_lins_long = []

    for (i, path) in enumerate(files):
        if keyword in path:
            file = np.load(path)
            pulse_width = file["pulse_length"] * 1e6  # in us
            pulse_power = file["laser_power"]  # in nW
            period = 1 / file['exc_pulse_freq'] * 1e3  # in ms
            bin_width = period / len(file['bins'])  # in ms
            start = file['hist'].argmax() + 5
            t_start = file['bins'][start]
            bins = file['bins'][start:]
            hist = file['hist'][start:]
            tot_count = sum(hist)
            PL_count = tot_count - len(bins) * floor_counts_per_bin
            hist = file['hist'][start:] - floor_counts_per_bin  # subtract dark counts for each bin
            hist = np.where(hist < 0, 0, hist)
            floor_counts_in_hist = floor_counts_per_bin * len(bins)
            actual_freq = np.average(file['freq'])
            laser_wl = np.average(file['laser_wls'])
            powers.append(pulse_power)
            widths.append(pulse_width)
            #         tot_counts.append(tot_count-floor_counts_in_hist if tot_count-floor_counts_in_hist > 0 else 0)
            PL_counts.append(PL_count)
            actual_freqs.append(actual_freq)
            laser_wls.append(laser_wl)
            # rebin  1000 to 100 bins
            # factor = 15
            new_bin_width = factor * bin_width
            new_bins = []
            new_hist = []
            for j in range(int(len(hist) / factor)):
                index = j * factor
                new_bins.append(bins[index])
                new_hist.append(sum(hist[index:index + factor]))
            new_bins = np.array(new_bins)
            new_hist = np.array(new_hist)
            #             print(len(new_bins))
            # new_hist = where(new_hist<0, 0, new_hist)
            bin_width = new_bin_width
            x = new_bins[np.nonzero(new_hist)]
            y = new_hist[np.nonzero(new_hist)]
            # print(new_hist, len(x), len(y))
            #             subplot(num_files, 1, i+1)
            x = new_bins[np.nonzero(new_hist)]
            y = new_hist[np.nonzero(new_hist)]
            # print(new_hist, len(x), len(y))

    return x, y


# get data from files
x, y = draw_hist_short_long(DATA_PATH, keyword="", factor=1, floor_counts_per_bin=0)

# do fitting
# popt, pcov = curve_fit(exponential_const, x, y, p0=[8000, 2, 18000])
model = ConstantModel() + ExponentialModel()
res = model.fit(y, x=x,
                amplitude=8000, decay=2, c=18000)

plt.plot(x, y,
         color=color, label='Data')
plt.plot(x, res.best_fit,
         '--k', label='Fit')
         # label='$T_{1}$=' + f'{round(popt[1], 2)}' + '$\pm$' + f'{round(np.sqrt(pcov[1][1]), 4)}' + ' ms')

plt.text(max(x)/2, (max(y)+min(y))/2,
         rf"$T_1 = {res.params['decay'].value:.3} \pm {res.params['decay'].stderr:.3}$ ms")

plt.title(r'Er$^{3+}$:MgLiNb, Amplitude Damping')
plt.ylabel('PL + Dark Counts')
plt.xlabel('Time (ms)')
plt.legend(shadow=True)
plt.grid('on')

plt.tight_layout()
plt.show()
