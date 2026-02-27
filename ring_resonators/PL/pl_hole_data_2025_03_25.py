import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import ExponentialModel, ConstantModel


HOLE_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
             "/New_mounted_device/10mK/pl_holeburn_2025_03_25/test_locked5.npy")

# data params
integration_time = 300  # unit: s
pl_freq = 20  # unit: Hz
pl_period = (1/pl_freq)*1e3  # unit: ms

# processing params
idx_to_skip = 5
smoothing = 20  # unit: * pl_period

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
color2 = 'coral'
color3 = 'mediumpurple'


# get data
data = np.load(HOLE_DATA)
pl_data = np.sum(data, axis=0)
pl_time = np.linspace(0, pl_period, num=len(pl_data))
counts_data = np.sum(data, axis=1)
counts_time = np.linspace(0, integration_time, num=len(counts_data))
counts_smoothed = np.convolve(counts_data, np.ones(smoothing)/smoothing, mode='valid')
counts_time_smoothed = counts_time[:-(smoothing-1)]


plt.plot(pl_time[idx_to_skip:], pl_data[idx_to_skip:],
         color=color)

plt.tight_layout()
plt.show()


plt.plot(counts_time_smoothed, counts_smoothed,
         color=color2)
# plt.xlim(-10, 30)
# plt.xlim(200, 280)

plt.tight_layout()
plt.show()


# plot with no labels
plt.pcolormesh(data, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()


# plot with labels
X, Y = np.meshgrid(pl_time, counts_time)
plt.pcolormesh(X, Y, data, cmap='gray')
plt.xlabel('PL Time (ms)')
plt.ylabel('Integration Time (s)')

plt.tight_layout()
plt.show()
