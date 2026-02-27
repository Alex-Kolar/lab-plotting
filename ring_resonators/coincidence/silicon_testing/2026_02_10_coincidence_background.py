import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit import Model
from lmfit.models import GaussianModel

from ring_resonators.cavity_fit.cavity_metrics import g_2_single_exp


DATA_DIR = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
            '/Silicon_test_devices/mk_5/chip_2/2026_02_10/characterize_background')
FILE_FMT = 'correlation_550mA_{}mA_1m.txt'
POWER_DICT = {700: 18.2,  # map amplifier setpoint (in mA) to measured output power (in uW)
              650: 14.2,
              600: 10.5,
              550: 7.4,
              500: 4.7,
              450: 2.9,
              400: 1.3,
              350: 0.23}
coincidence_center_guess = -0.01  # unit: us
fit_range_size = 0.2  # unit: us

# plotting params
OUTPUT_DIR = ('/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators/silicon_testing'
              '/silicon_mk_5/chip_2/pair_generation/pulse_power_sweep_2026_02_10/no_fit_background')
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    os.makedirs(os.path.join(OUTPUT_DIR, 'background_fits'))
    os.makedirs(os.path.join(OUTPUT_DIR, 'peak_fits'))

dfs = {}
bg_model = GaussianModel()
peak_model = Model(g_2_single_exp)
all_res = []
all_res_bg = []
for key, val in POWER_DICT.items():
    df = pd.read_csv(os.path.join(DATA_DIR, FILE_FMT.format(key)), sep='\t')
    dfs[key] = df

    # get background
    lower_bound = int(coincidence_center_guess*1e6 + 2e6 - (fit_range_size*1e6/2))
    upper_bound = int(coincidence_center_guess*1e6 + 2e6 + (fit_range_size*1e6/2))
    idx_bg = np.logical_and(df['Time(ps)'] > lower_bound, df['Time(ps)'] < upper_bound)
    time_bg = df['Time(ps)'][idx_bg].to_numpy() / 1e6  # convert to us
    counts_bg = df['Counts'][idx_bg].to_numpy()

    plt.plot(time_bg, counts_bg,
             color='coral')

    plt.title(f'Background for Output Power {val} uW')
    plt.xlabel(r'Time ($\mathrm{\mu}$s)')
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'background_fits', f'background_fit_{key}mA.png'))
    plt.clf()

    # fit peak
    lower_bound = int(coincidence_center_guess*1e6 - (fit_range_size*1e6 / 2))
    upper_bound = int(coincidence_center_guess*1e6 + (fit_range_size*1e6 / 2))
    idx_to_fit = np.logical_and(df['Time(ps)'] > lower_bound, df['Time(ps)'] < upper_bound)
    time_to_fit = df['Time(ps)'][idx_to_fit].to_numpy() / 1e6  # convert to us
    counts_to_fit = df['Counts'][idx_to_fit].to_numpy()
    counts_no_bg = counts_to_fit - counts_bg  # subtract background fit
    amplitude_guess = max(counts_no_bg)
    tau_guess = 1e-3
    peak_res = peak_model.fit(counts_no_bg, x=time_to_fit,
                              x0=coincidence_center_guess,
                              amplitude=amplitude_guess,
                              tau=tau_guess)
    all_res.append(peak_res)

    plt.plot(time_to_fit, counts_to_fit,
             color='coral', label='Data')
    plt.plot(time_to_fit, counts_bg,
             ls='--', color='gray', label='Background Data')
    plt.plot(time_to_fit, peak_res.best_fit+counts_bg,
             ls='--', color='k', label='Fit')

    plt.title(f'Coincidence Peak for Output Power {val} uW')
    plt.xlabel(r'Time ($\mathrm{\mu}$s)')
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'peak_fits', f'peak_fit_{key}mA.png'))
    plt.clf()

    plt.plot(time_to_fit, counts_no_bg,
             color='mediumpurple', label='Data')
    plt.plot(time_to_fit, peak_res.best_fit,
             ls='--', color='k', label='Fit')

    plt.title(f'Coincidence Peak for Output Power {val} uW (with Background Removed)')
    plt.xlabel(r'Time ($\mathrm{\mu}$s)')
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'peak_fits', f'peak_fit_no_background_{key}mA.png'))
    plt.clf()

# # plot final collected metrics
#
# # plot of fitted background pulse
# all_bg_height = [res.params['height'].value for res in all_res_bg]
# all_bg_height_err = [res.params['height'].stderr for res in all_res_bg]
# all_bg_fwhm = [res.params['fwhm'].value*1e3 for res in all_res_bg]  # convert to ns
# all_bg_fwhm_err = [res.params['fwhm'].stderr*1e3 for res in all_res_bg]
# fig, ax = plt.subplots()
# ax2 = ax.twinx()
# ax.errorbar(list(POWER_DICT.values()), all_bg_height, yerr=all_bg_height_err,
#             marker='o', ls='', capsize=3, color='cornflowerblue')
# ax2.errorbar(list(POWER_DICT.values()), all_bg_fwhm, yerr=all_bg_fwhm_err,
#              marker='o', ls='', capsize=3, color='coral')
# ax.set_xlabel(r'Output Power ($\mathrm{\mu}$W)')
# ax.set_ylabel(r'Pulse Fit Height (counts)', color='cornflowerblue')
# ax2.set_ylabel(r'Pulse Fit FWHM (ns)', color='coral')
# ax2.set_ylim(0, 50)
# fig.tight_layout()
# fig.show()
#
# # plot of g(2) information
# all_peak_height = [res.params['amplitude'].value for res in all_res]
# all_peak_height_err = [res.params['amplitude'].stderr for res in all_res]
# all_g2 = np.array(all_peak_height)/np.array(all_bg_height) + 1
#
# plt.errorbar(list(POWER_DICT.values()), all_g2,
#              ls='', marker='o', capsize=3, color='cornflowerblue')
#
# plt.xlabel(r'Output Power ($\mathrm{\mu}$W)')
# plt.ylabel(r'$g^{(2)}(0)$')
# plt.xscale('log')
# plt.yscale('log')
# plt.tight_layout()
# plt.show()
