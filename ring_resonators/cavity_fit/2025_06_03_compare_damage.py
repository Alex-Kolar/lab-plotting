import glob
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import BreitWignerModel, LinearModel
from scipy.signal import find_peaks
import pickle


DATA_DIR_INIT = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
                 "/New_mounted_device/10mK/03262025")
CSV_INIT = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/10mK/03262025/resonances_03_26_2025.csv")
DATA_DIR_POST = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
                 "/New_mounted_device/10mK/06032025")
CSV_POST = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/10mK/06032025/resonances_06_03_2025.csv")

# format: first row has init files, second row has post files
FILES_TO_COMPARE = ((3, 9),
                    (1, 4))
LABELS = ["Signal Resonance", "Pump Resonance"]

AOM_OFFSET_INIT = 0.600
AOM_OFFSET_POST = 0.400

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color_init = 'cornflowerblue'
color_post = 'coral'


# gather data
# read csv
main_df_init = pd.read_csv(CSV_INIT)
main_df_post = pd.read_csv(CSV_POST)


# fit data
data_dfs = {'init': [], 'post': []}
freqs = {'init': [], 'post': []}

for file_no in FILES_TO_COMPARE[0]:
    filename = f'SDS{file_no:05}.csv'
    filepath = os.path.join(DATA_DIR_INIT, filename)
    data_dfs['init'].append(pd.read_csv(filepath, header=10, skiprows=[11]))

    row_idx = main_df_init['FileNumber'] == file_no
    freq_min = float(main_df_init['Min'][row_idx].iloc[0])
    freq_max = float(main_df_init['Max'][row_idx].iloc[0])
    freq_min += AOM_OFFSET_INIT
    freq_max += AOM_OFFSET_INIT
    freqs['init'].append((freq_min, freq_max))

for file_no in FILES_TO_COMPARE[1]:
    filename = f'SDS{file_no:05}.csv'
    filepath = os.path.join(DATA_DIR_POST, filename)
    data_dfs['post'].append(pd.read_csv(filepath, header=10, skiprows=[11]))

    row_idx = main_df_post['FileNumber'] == file_no
    freq_min = float(main_df_post['Min'][row_idx].iloc[0])
    freq_max = float(main_df_post['Max'][row_idx].iloc[0])
    freq_min += AOM_OFFSET_POST
    freq_max += AOM_OFFSET_POST
    freqs['post'].append((freq_min, freq_max))


# plotting
for init_df, post_df, init_freq, post_freq, label \
        in zip(data_dfs['init'], data_dfs['post'], freqs['init'], freqs['post'], LABELS):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    start = min(init_freq[0], post_freq[0])

    ramp = init_df['CH1'].astype(float)
    transmission = init_df['CH2'].astype(float)
    id_min = np.argmin(ramp)
    id_max = np.argmax(ramp)
    transmission = transmission[id_min:id_max]
    transmission.reset_index(drop=True, inplace=True)
    freq = np.linspace(init_freq[0]-start, init_freq[1]-start, num=(id_max-id_min))

    ln1 = ax1.plot(freq, transmission, color=color_init, label='Before Pumping')
    ax1.set_ylim(0, 1.1*max(transmission))

    ramp = post_df['CH1'].astype(float)
    transmission = post_df['CH2'].astype(float)
    id_min = np.argmin(ramp)
    id_max = np.argmax(ramp)
    transmission = transmission[id_min:id_max]
    transmission.reset_index(drop=True, inplace=True)
    freq = np.linspace(post_freq[0]-start, post_freq[1]-start, num=(id_max - id_min))

    ln2 = ax2.plot(freq, transmission, color=color_post, label='After Pumping')
    ax2.set_ylim(0, 1.1 * max(transmission))

    ax1.set_title(label)
    ax1.set_xlabel(f'Detuning from {start:.0f} (GHz)')
    ax1.set_ylabel('Transmission Before Pumping (A.U.)')
    ax2.set_ylabel('Transmission After Pumping (A.U.)')
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)

    fig.tight_layout()
    fig.show()
