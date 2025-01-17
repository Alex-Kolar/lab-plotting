import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


FILTER_HIGH = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
               "/filter_data/DWDM_CH_48/room_temp_scan_01092025/FILTERHIGH.csv")
FILTER_LOW = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
              "/filter_data/DWDM_CH_48/room_temp_scan_01092025/FILTERLOW.csv")
FILTER_MID = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
              "/filter_data/DWDM_CH_48/room_temp_scan_01092025/FILTERMID.csv")
FREQ_HIGH = (194810.449, 194841.492)
FREQ_LOW = (194748.541, 194779.548)
FREQ_MID = (194777.650, 194808.516)
FREQ_CENTER = 194800.000

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'mediumpurple'


all_paths = [FILTER_HIGH, FILTER_LOW, FILTER_MID]
all_ranges = [FREQ_HIGH, FREQ_LOW, FREQ_MID]
freqs = []
trans = []
for path, range in zip(all_paths, all_ranges):
    freq_start, freq_end = range
    data_df = pd.read_csv(path, header=10, skiprows=[11])

    ramp = data_df['CH1'].astype(float)
    transmission = data_df['CH2'].astype(float)

    id_min = np.argmin(ramp)
    id_max = np.argmax(ramp)
    transmission = transmission[id_min:id_max]
    transmission.reset_index(drop=True, inplace=True)
    freq = np.linspace((freq_start - FREQ_CENTER), (freq_end - FREQ_CENTER),
                       num=(id_max - id_min))  # unit: GHz

    freqs.append(freq)
    trans.append(transmission)


# plotting of data
for freq, tran in zip(freqs, trans):
    plt.plot(freq, tran, color=color)

plt.title("Filter Transmission Test")
plt.xlabel("Detuning from 194800 (GHz)")
plt.ylabel("Transmission (A.U.)")
plt.grid(True)

plt.tight_layout()
plt.show()
