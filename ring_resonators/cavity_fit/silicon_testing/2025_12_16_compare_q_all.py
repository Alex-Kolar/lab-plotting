import os
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


INPUT_DIRS = ('/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators/silicon_testing/silicon_mk_2',)
# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
color_contrast = 'coral'


# read q data
input_data = {'q': {},
              'centers': {},
              'contrasts': {}}
for dir in INPUT_DIRS:
    with open(os.path.join(dir, 'res_data.bin'), 'rb') as f:
        data = pickle.load(f)
        input_data['q'] = input_data['q'] | data['q']
        input_data['centers'] = input_data['centers'] | data['centers']
        input_data['contrasts'] = input_data['contrasts'] | data['contrasts']


# make bar graph of highest q
q_data = input_data['q']
bar_x = range(len(q_data.keys()))
labels = list(q_data.keys())
bar_y = [max(vals) for vals in q_data.values()]
labels, bar_y = zip(*sorted(zip(labels, bar_y)))

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(bar_x, bar_y,
       color=color, edgecolor='k', zorder=2)

ax.set_xticks(bar_x, labels)
ax.set_xlabel("Device Number")
ax.set_ylabel("Highest Q Factor")
ax.grid(axis='y')

fig.tight_layout()
fig.show()


# make bar graph of contrast for highest q
contrast_data = input_data['contrasts']
bar_x = range(len(contrast_data.keys()))
labels = list(contrast_data.keys())

# TODO: finish this to make the contrast correspond to highest Q
bar_y = []
for qs, contrasts in zip(q_data.values(), contrast_data.values()):
    contrast_idx = np.argmax(qs)
    bar_y = 0
# contrast_idx = [np.argmax(q_data) for q in q_data.values()]
# bar_y = [vals[contrast_] for vals in contrast_data.values()]
labels, bar_y = zip(*sorted(zip(labels, bar_y)))

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(bar_x, bar_y,
       color=color_contrast, edgecolor='k', zorder=2)

ax.set_xticks(bar_x, labels)
ax.set_xlabel("Device Number")
ax.set_ylabel("Highest Contrast")
ax.grid(axis='y')

fig.tight_layout()
fig.show()
