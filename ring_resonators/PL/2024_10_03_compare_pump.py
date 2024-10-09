import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle


DATA_PRE = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
            "/new_mounted/10mK_pl/all_fitted_decay/10022024/res_data.bin")
DATA_POST = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
             "/new_mounted/10mK_pl/all_fitted_decay/10032024/res_data.bin")

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})


with open(DATA_PRE, "rb") as f:
    data_pre = pickle.load(f)
with open(DATA_POST, "rb") as f:
    data_post = pickle.load(f)

# keep only relevant data
freqs_pre = data_pre['freqs'] + data_pre['freq_min']
freqs_post = data_post['freqs'] + data_post['freq_min']
idx_to_keep = np.where(freqs_pre - min(freqs_post) > 0)[0]
freq_diff = data_post['freq_min'] - data_pre['freq_min']


# plotting
fig, ax_pre = plt.subplots()
ax_post = ax_pre.twinx()

ax_pre.errorbar(data_pre["freqs"][idx_to_keep] - freq_diff,
                data_pre["area_fit"][0][idx_to_keep], data_pre["area_fit"][1][idx_to_keep],
                ls='', marker='o', capsize=3, color='cornflowerblue')
ax_post.errorbar(data_post["freqs"],
                 data_post["area_fit"][0], data_post["area_fit"][1],
                 ls='', marker='s', capsize=3, color='coral')

ax_pre.set_xlabel(f'Frequency + {data_pre['freq_min']:.3f} (GHz)')
ax_pre.set_ylabel('Fitted PL Area Before Pumping (A.U.)',
                  color='cornflowerblue')
ax_post.set_ylabel('Fitted PL Area After Pumping (A.U.)',
                   color='coral')
ax_pre.grid(True)
ax_pre.set_ylim((0, 1.2))

fig.tight_layout()
fig.show()
