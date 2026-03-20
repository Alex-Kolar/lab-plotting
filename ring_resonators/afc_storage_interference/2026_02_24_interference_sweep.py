import glob
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.ticker as tck
import matplotlib.pyplot as plt
from lmfit.models import SineModel, ConstantModel


DATA_DIR = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
            '/Mounted_device_mk_5/10mK/2026_02_24/afc/afc_dual_comb_no_phase_offset')  # contains sweep files
one_pulse_offres = 'one_pulse_offres.txt'
one_pulse_storage = 'one_pulse_3_00pi.txt'
two_pulse_offres = 'two_pulse_offres.txt'
interference_window = (1.97-0.05, 1.97+0.05)

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
PLOT_PEAKS = False
xlim = (1.97-0.2, 1.97+0.2)
OUTPUT_DIR = ('/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators'
              '/mounted_mk_5/10mK_echo/interference_sweep/no_phase_offset')


# gather data
storage_files = glob.glob('two_pulse_*pi.txt', root_dir=DATA_DIR)
storage_files.sort()

dfs = []
all_phases = []
all_interference_counts = []
for file in storage_files:
    file_path = os.path.join(DATA_DIR, file)
    df = pd.read_csv(file_path, sep='\t')
    dfs.append(df)

    file_parts = file.split('_')
    pi_fraction = float(file_parts[2]) + (float(file_parts[3][:2])/100)
    all_phases.append(pi_fraction)

    # plot interference
    time = df['time(ps)']
    time /= 1e6  # convert to us
    counts = df['counts']
    plt.plot(time, counts, color='cornflowerblue')
    plt.title(rf'Comb Separation $\Delta\phi = {pi_fraction}\pi$')
    plt.ylim(0, 800)
    plt.xlim(xlim)
    plt.xlabel(r'Time ($\mu$s)')
    plt.ylabel(r'Counts')
    plt.tight_layout()

    file_base = os.path.splitext(file)[0]
    output_filename = file_base + '.png'
    plt.savefig(os.path.join(OUTPUT_DIR, output_filename))
    plt.clf()

    # gather relevant data
    int_idx = np.where(np.logical_and(time > interference_window[0], time < interference_window[1]))[0]
    interference_counts = np.sum(counts[int_idx])
    all_interference_counts.append(interference_counts)


# fitting
model = SineModel() + ConstantModel()
res = model.fit(all_interference_counts, x=all_phases,
                amplitude=4500, frequency=np.pi, c=4500)
print(res.fit_report())
amplitude = res.params['amplitude'].value
amplitude_err = res.params['amplitude'].stderr
constant = res.params['c'].value
constant_err = res.params['c'].stderr
visibility = amplitude/constant
visibility_err = visibility * np.sqrt((amplitude_err/amplitude)**2 + (constant_err/constant)**2)

print(f'Visibility: {visibility*100:.2f} +/- {visibility_err*100:.2f}%')

# final plotting
x_points_for_eval = np.linspace(min(all_phases), max(all_phases), 1000)
plt.plot(all_phases, all_interference_counts,
         color='cornflowerblue', ls='', marker='o',
         label='Data')
plt.plot(x_points_for_eval, res.eval(x=x_points_for_eval),
         color='cornflowerblue', ls='--',
         label='Fit')
plt.xlabel(r'Comb Phase Difference $\Delta\phi$')
plt.ylabel('Counts')
plt.legend(framealpha=1)

ax = plt.gca()
ax.xaxis.set_major_formatter(tck.FormatStrFormatter(r'%g$\pi$'))
ax.xaxis.set_major_locator(tck.MultipleLocator(base=1.0))

plt.tight_layout()
plt.show()
