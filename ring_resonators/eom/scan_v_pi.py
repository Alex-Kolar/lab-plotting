import glob
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/eom_data/07292024/vpi_scan")

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})


# find and read spectrum analyzer files
filenames = glob.glob('*.csv', root_dir=DATA_DIR)
v_pis = []
data_dfs = []
for file in filenames:
    file_str = os.path.splitext(file)[0]
    file_str = file_str[:-1]  # remove 'v'
    v_pi_str = file_str.replace('p', '.')
    v_pis.append(float(v_pi_str))
    file_path = os.path.join(DATA_DIR, file)
    data_dfs.append(pd.read_csv(file_path, header=1))

# sort
v_pis, data_dfs = zip(*sorted(zip(v_pis, data_dfs)))


X, Y = np.meshgrid(data_dfs[0]['Freq'], v_pis)
X = X.astype(float)
X /= 1e6  # convert to MHz
Z = np.zeros_like(X)
for i, df in enumerate(data_dfs):
    Z[i, :] = df['Amp']

# plotting
plt.pcolormesh(X, Y, Z)
plt.xlabel('Frequency (MHz)')
plt.ylabel(r'Guess for $V_\pi$ (V)')
plt.colorbar(label='Power (dBm)')

plt.tight_layout()
plt.show()


# plot max
maxs = [max(data_df['Amp']) for data_df in data_dfs]
plt.plot(v_pis, maxs,
         'o-')
plt.xlabel(r'Guess for $V_\pi$ (V)')
plt.ylabel('Maximum Power (dBm)')

plt.tight_layout()
plt.show()


# plot all
v_pi_to_plot = [3.0, 4.2, 4.4]
idx = [v_pis.index(v_pi) for v_pi in v_pi_to_plot]
for v_pi, i in zip(v_pi_to_plot, idx):
    plt.plot(data_dfs[i]['Freq'] / 1e6,
             data_dfs[i]['Amp'],
             label=rf'$V_\pi$ = {v_pi}')
plt.legend(shadow=True)
plt.xlabel(r'Frequency (MHz)')
plt.ylabel('Power (dBm)')
plt.grid(True)

plt.tight_layout()
plt.show()
