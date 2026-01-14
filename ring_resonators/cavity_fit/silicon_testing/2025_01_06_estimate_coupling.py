import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


INPUT_DATA = ('/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators'
              '/silicon_testing/silicon_mk_3/f_etch/res_data.bin',
              '/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators'
              '/silicon_testing/silicon_mk_3/f_etch/res_data_2.bin')
DEV_TO_KEEP = [7, 8, 9, 11, 12, 13]
# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color_q = 'cornflowerblue'
color_c = 'coral'
width = 0.4


# read data
input_data = {'q': {},
              'contrasts': {},
              'centers': {}}
for data in INPUT_DATA:
    with open(data, 'rb') as f:
        data = pickle.load(f)
        input_data['q'] = input_data['q'] | data['q']
        input_data['contrasts'] = input_data['contrasts'] | data['contrasts']
        input_data['centers'] = input_data['centers'] | data['centers']

# remove devices that we don't care about
for property in 'q', 'contrasts', 'centers':
    input_data[property] = {dev: input_data[property][dev] for dev in DEV_TO_KEEP}


# make bar graph of q and contrast
q_data = input_data['q']
c_data = input_data['contrasts']
bar_x = np.array(range(len(q_data.keys())))
labels = list(q_data.keys())
labels = sorted(labels)
bar_q = [max(q_data[label]) for label in labels]
bar_c = [max(c_data[label]) for label in labels]
# bar_q = [max(vals) for vals in q_data.values()]
# labels, bar_y = zip(*sorted(zip(labels, bar_y)))

fig, ax = plt.subplots(figsize=(10, 5))
ax_r = ax.twinx()

ax.bar(bar_x-(width/2), bar_q,
       color=color_q, edgecolor='k', width=width, zorder=2)
ax_r.bar(bar_x+(width/2), bar_c,
         color=color_c, edgecolor='k', width=width, zorder=2)

ax.set_xticks(bar_x, labels)
ax.set_xlabel("Device Number")
ax.set_ylabel("Highest Q Factor")
ax.grid(axis='y')

fig.tight_layout()
fig.show()
