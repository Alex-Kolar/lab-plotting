import os
import numpy as np
import glob
import pickle
from lmfit.models import ExponentialModel, ConstantModel
from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.pyplot as plt


# data
PL_DATA = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Lithium Niobate'
           '/New Devices/waveguide/4K/2025_11_10/pl_sweep')
OUTPUT_DIR = ('/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/lithium_niobate'
              '/new_devices/waveguide/4K_pl/2025_11_10/')
OUTPUT_DATA_FILE = 'res_data.bin'
AOM_OFFSET = 0.600  # unit: GHz

# meta idx params
CUTOFF_IDX = 5

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color_pl = 'coral'
color_lifetime = 'mediumpurple'
ref_freq = 195950  # unit: GHz
freq_range = (AOM_OFFSET, AOM_OFFSET + 3)

LOAD_PL = False  # If true, load and fit all files. If false, look for pre-fit data in OUTPUT_DIR.


# gather PL data
dump_file = os.path.join(OUTPUT_DIR, OUTPUT_DATA_FILE)

if LOAD_PL:
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    model = ExponentialModel() + ConstantModel()

    all_files = glob.glob('pl_experiment_*.npz', root_dir=PL_DATA)
    all_files = sorted(all_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    all_data = {'T1': [],
                'T1 errors': [],
                'Frequencies': [],
                'Frequency errors': [],
                'Areas': [],
                'Dark counts': [],
                'Dark count errors': []}

    num_files = len(all_files)
    num_digits = len(str(num_files))

    for i, filename in enumerate(all_files):
        padded_str = str(i).zfill(num_digits)
        print(f'Loading file {padded_str}/{num_files}', end='\r')

        file_base = os.path.splitext(filename)[0]
        file_num = int(file_base[14:])
        full_filename = os.path.join(PL_DATA, filename)

        data = np.load(full_filename)
        bins = data['bins'][CUTOFF_IDX:]
        counts = data['counts'][CUTOFF_IDX:]
        target_freq = data['target frequency']
        frequencies = data['measured frequencies']
        mean_freq = np.mean(frequencies)
        if abs(mean_freq - target_freq) > 0.05:
            print('')
            print(f'skipping file #{file_num} (invalid frequency)')
            continue

        all_data['Frequencies'].append(mean_freq + AOM_OFFSET)  # account for AOMs in path
        all_data['Frequency errors'].append(np.std(frequencies))

        decay_guess = 2.5
        amplitude_guess = counts[0]
        res = model.fit(counts, x=bins,
                        amplitude=amplitude_guess, decay=2.5)

        # extract data from fit
        t1 = res.params['decay'].value
        t1_err = res.params['decay'].stderr
        dark = res.params['c'].value
        dark_err = res.params['c'].stderr
        all_data['T1'].append(t1)
        all_data['T1 errors'].append(t1_err)
        all_data['Dark counts'].append(dark)
        all_data['Dark count errors'].append(dark_err)
        area = np.sum(counts, axis=0) - (dark*len(counts))
        all_data['Areas'].append(area)

        # plotting
        plt.plot(bins, counts, color=color_pl)
        plt.plot(bins, res.best_fit,
                 color='k', ls='--')
        text = rf'$T_1$ = {t1:.3f} $\pm$ {t1_err:.3f} ms'
        ax = plt.gca()
        plt.text(0.95, 0.95, text,
                 ha='right', va='top',
                 transform=ax.transAxes)
        plt.xlabel('Time (ms)')
        plt.ylabel('Counts')

        plt.tight_layout()
        save_name = f'pl_{file_num:03d}_{target_freq:.3f}'.replace('.', '_')
        plt.savefig(os.path.join(OUTPUT_DIR, save_name+'.png'))
        plt.clf()

    # dump data
    with open(dump_file, 'wb') as f:
        pickle.dump(all_data, f)

else:
    with open(dump_file, 'rb') as f:
        all_data = pickle.load(f)


freqs = np.array(all_data['Frequencies']) - ref_freq
area = all_data['Areas']
area_err = all_data['Dark count errors']
t1 = np.array(all_data['T1'])
t1_err = all_data['T1 errors']

# plotting
fig, axs = plt.subplots(2, 1, sharex=True)

axs[0].errorbar(freqs, area, yerr=area_err,
                color=color_pl, capsize=3, ls='', marker='o')
axs[1].errorbar(freqs, t1, yerr=t1_err,
                color=color_lifetime, capsize=3, ls='', marker='o',
                label=r'$T_1$ Data')
# axs[0].set_xlim(freq_range)
axs[1].set_ylim(0, 10)
axs[0].set_ylabel('PL Area')
axs[1].set_ylabel('PL Lifetime (ms)')
axs[-1].set_xlabel(f'Detuning (GHz) from {ref_freq:.0f} GHz')
axs[0].set_title('Lithium Niobate PLE Scan')

fig.tight_layout()
fig.show()
