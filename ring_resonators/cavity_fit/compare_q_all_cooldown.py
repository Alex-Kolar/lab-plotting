import os
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


INPUT_DIRS_WARM = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
                   "/new_mounted/room_temp_cavity/room_temp_07182024",
                   "/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
                   "/new_mounted/room_temp_cavity/room_temp_07192024",
                   "/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
                   "/new_mounted/room_temp_cavity/room_temp_07222024")
INPUT_DIRS_COLD = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
                   "/new_mounted/4K_cavity/4K_07252024/testing_3",
                   "/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
                   "/new_mounted/4K_cavity/4K_09262024/testing",)
# plotting params
figsize = (12, 4)
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color_warm = 'coral'
color_cold = 'cornflowerblue'
width = 0.8
device_range = (4, 19)
SAVE_FIG = False
OUTPUT_DIR = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
              "/new_mounted/all_q_temperatures.svg")


# read q data
input_data = {'q': {},
              'centers': {}}
for dir in INPUT_DIRS_WARM:
    with open(os.path.join(dir, 'res_data.bin'), 'rb') as f:
        data = pickle.load(f)
        input_data['q'] = input_data['q'] | data['q']
        input_data['centers'] = input_data['centers'] | data['centers']
input_data_cold = {'q': {},
                   'centers': {}}
for dir in INPUT_DIRS_COLD:
    with open(os.path.join(dir, 'res_data.bin'), 'rb') as f:
        data = pickle.load(f)
        input_data_cold['q'] = input_data_cold['q'] | data['q']
        input_data_cold['centers'] = input_data_cold['centers'] | data['centers']



# make bar graph of highest q
q_data = input_data['q']
loc = np.fromiter(q_data.keys(), dtype=int)
bar_y = [max(vals) for vals in q_data.values()]
q_data_cold = input_data_cold['q']
loc_cold = np.fromiter(q_data_cold.keys(), dtype=int)
bar_y_cold = [max(vals) for vals in q_data_cold.values()]
ticks = np.arange(device_range[0], device_range[1]+1, 1)

fig, ax = plt.subplots(figsize=figsize)

ax.bar(loc-(width/4), bar_y,
       color=color_warm, edgecolor='k', zorder=2, width=width/2,
       label=r'$T$ = 300 K')
ax.bar(loc_cold+(width/4), bar_y_cold,
       color=color_cold, edgecolor='k', zorder=2, width=width/2,
       label=r'$T$ = 4 K')

ax.set_title("Q Factor by Device")
ax.set_xlabel("Device Number")
ax.set_ylabel("Highest Q Factor")
ax.grid(axis='y')
ax.legend(shadow=True)
ax.set_xticks(ticks)

fig.tight_layout()
if SAVE_FIG:
    fig.savefig(OUTPUT_DIR)
else:
    fig.show()
