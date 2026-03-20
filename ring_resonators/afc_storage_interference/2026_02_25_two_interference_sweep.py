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
DATA_DIR_PHASE = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
                  '/Mounted_device_mk_5/10mK/2026_02_24/afc/afc_dual_comb_half_pi_phase_offset')
one_pulse_offres = 'one_pulse_offres.txt'
one_pulse_storage = 'one_pulse_3_00pi.txt'
two_pulse_offres = 'two_pulse_offres.txt'
interference_window = (1.99-0.05, 1.99+0.05)

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
xlim = (1.97-0.2, 1.97+0.2)
OUTPUT_DIR = ('/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators'
              '/mounted_mk_5/10mK_echo/interference_sweep/no_phase_offset')
OUTPUT_DIR_PHASE = ('/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators'
                    '/mounted_mk_5/10mK_echo/interference_sweep/phase_offset')
color_no_phase = 'cornflowerblue'
color_phase = 'coral'


# gather data
storage_files = glob.glob('two_pulse_*pi.txt', root_dir=DATA_DIR)
storage_files.sort()
storage_files_phase = glob.glob('two_pulse_*pi.txt', root_dir=DATA_DIR_PHASE)
storage_files_phase.sort()

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
    plt.plot(time, counts, color=color_no_phase)
    plt.axvspan(interference_window[0], interference_window[1],
                alpha=0.2, color='gray', label='Interference Count Window')
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

dfs_phase = []
all_phases_phase = []
all_interference_counts_phase = []
for file in storage_files_phase:
    file_path = os.path.join(DATA_DIR_PHASE, file)
    df = pd.read_csv(file_path, sep='\t')
    dfs_phase.append(df)

    file_parts = file.split('_')
    pi_fraction = float(file_parts[4]) + (float(file_parts[5][:2])/100)
    all_phases_phase.append(pi_fraction)

    # plot interference
    time = df['time(ps)']
    time /= 1e6  # convert to us
    counts = df['counts']
    plt.plot(time, counts, color=color_phase)
    plt.axvspan(interference_window[0], interference_window[1],
                alpha=0.2, color='gray', label='Interference Count Window')
    plt.title(rf'Comb Separation $\Delta\phi = {pi_fraction}\pi$')
    plt.ylim(0, 800)
    plt.xlim(xlim)
    plt.xlabel(r'Time ($\mu$s)')
    plt.ylabel(r'Counts')
    plt.tight_layout()

    file_base = os.path.splitext(file)[0]
    output_filename = file_base + '.png'
    plt.savefig(os.path.join(OUTPUT_DIR_PHASE, output_filename))
    plt.clf()

    # gather relevant data
    int_idx = np.where(np.logical_and(time > interference_window[0], time < interference_window[1]))[0]
    interference_counts = np.sum(counts[int_idx])
    all_interference_counts_phase.append(interference_counts)


# fitting
model = SineModel() + ConstantModel()
params = model.make_params(amplitude=4500, frequency=np.pi, c=4500)
params['frequency'].vary = False

res = model.fit(all_interference_counts, params, x=all_phases)
print(res.fit_report())

amplitude = res.params['amplitude'].value
amplitude_err = res.params['amplitude'].stderr
constant = res.params['c'].value
constant_err = res.params['c'].stderr
visibility = amplitude/constant
visibility_err = visibility * np.sqrt((amplitude_err/amplitude)**2 + (constant_err/constant)**2)
print(f'Visibility: {visibility*100:.2f} +/- {visibility_err*100:.2f}%')
print(f'Fidelity: {(visibility+1)/2:.4f} +/- {visibility_err/2:.4f}')
maximum = (np.pi/2 - res.params['shift']) / res.params['frequency']
print(f'Calculated Maximum Point: {maximum}')

res_phase = model.fit(all_interference_counts_phase, params, x=all_phases_phase)
print(res_phase.fit_report())

amplitude = res_phase.params['amplitude'].value
amplitude_err = res_phase.params['amplitude'].stderr
constant = res_phase.params['c'].value
constant_err = res_phase.params['c'].stderr
visibility_phase = amplitude/constant
visibility_phase_err = visibility * np.sqrt((amplitude_err/amplitude)**2 + (constant_err/constant)**2)
print(f'Visibility: {visibility_phase*100:.2f} +/- {visibility_phase_err*100:.2f}%')
print(f'Fidelity: {(visibility_phase+1)/2:.4f} +/- {visibility_phase_err/2:.4f}')
maximum = (np.pi/2 - res_phase.params['shift']) / res_phase.params['frequency']
print(f'Calculated Maximum Point: {maximum}')

# final plotting of points and fitted fringe
x_points_for_eval = np.linspace(min(all_phases), max(all_phases), 1000)
plt.plot(all_phases, all_interference_counts,
         color=color_no_phase, ls='', marker='o',
         label=rf'$|e\rangle + |\ell\rangle$')
plt.plot(x_points_for_eval, res.eval(x=x_points_for_eval),
         color=color_no_phase, ls='--',
         label='Fit')
plt.plot(all_phases_phase, all_interference_counts_phase,
         color=color_phase, ls='', marker='s',
         label=rf'$|e\rangle + i|\ell\rangle$')
plt.plot(x_points_for_eval, res_phase.eval(x=x_points_for_eval),
         color=color_phase, ls='--',
         label='Fit')
plt.title('Two-Comb Interference with Time Bin Input')
plt.xlabel(r'Comb Phase Difference $\Delta\phi$')
plt.ylabel('Counts')
plt.legend(framealpha=1)

ax = plt.gca()
ax.xaxis.set_major_formatter(tck.FormatStrFormatter(r'%g$\pi$'))
ax.xaxis.set_major_locator(tck.MultipleLocator(base=1.0))

plt.tight_layout()
plt.show()
