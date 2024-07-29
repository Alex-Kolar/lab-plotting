import os
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt


INPUT_DIRS = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
              "/new_mounted/room_temp_07182024",
              "/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
              "/new_mounted/room_temp_07192024")
# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'


# read q data
input_data = {'q': {},
              'centers': {}}
for dir in INPUT_DIRS:
    with open(os.path.join(dir, 'res_data.bin'), 'rb') as f:
        data = pickle.load(f)
        input_data['q'] = input_data['q'] | data['q']
        input_data['centers'] = input_data['centers'] | data['centers']

print(input_data)


# make bar graph of highest q
q_data = input_data['q']
bar_x = range(len(q_data.keys()))
labels = list(q_data.keys())
bar_y = [max(vals) for vals in q_data.values()]
labels, bar_y = zip(*sorted(zip(labels, bar_y)))

plt.bar(bar_x, bar_y,
        color=color, edgecolor='k', zorder=2)

plt.xticks(bar_x, labels)
plt.xlabel("Device Number")
plt.ylabel("Highest Q Factor")
plt.grid(axis='y')

plt.tight_layout()
plt.show()
