import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# experimental parameters
data_filename_1 = 'data/hom_2_13/hom_experiment_-5_5_mm.csv'
data_filename_2 = 'data/hom_2_24/hom_experiment_-5_5.csv'
num_bins = 1000
servo_pos_1 = np.arange(-5, 5, 0.005)
servo_pos_2 = np.arange(-5, 5, 0.01)

# plotting params
output_filename = 'output_figs/hom_-5_5_mm_combined.png'
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
x_range = (1.5, 3)  # unit: mm
# x_range = (1.5, 3)  # unit: mm
# x_range = (12, 19)  # unit: ps
y_range = (0, 2000)

columns = ["bin {}".format(ns) for ns in range(num_bins)]
df_1 = pd.read_csv(data_filename_1, names=columns)
df_2 = pd.read_csv(data_filename_2, names=columns)

# get bin for max coincidence counts
row = df_1.iloc[[0]]
bin_max = row.idxmax(axis=1)
coincidence_1 = df_1[bin_max].to_numpy()

row = df_2.iloc[[0]]
bin_max = row.idxmax(axis=1)
coincidence_2 = df_2[bin_max].to_numpy()

# convert displacement to timing
# relative_timing = (servo_pos * 1e-3) / 3e8
# relative_timing *= 1e12  # convert to ps
# relative_timing *= 2  # multiply by 2, since servo offset is doubled in path length

# fitting
# coincidence_norm = coincidence / np.max(coincidence)
# print(len(coincidence_norm))
# center_guess = relative_timing[np.argmin(coincidence_norm)]
# peak = GaussianModel()
# background = LinearModel()
# total_model = peak + background
# result = total_model.fit(1-coincidence_norm, x=relative_timing,
#                          center=center_guess)
# print(result.fit_report())

plt.plot(servo_pos_1, coincidence_1, 'o', label='February 13')
plt.plot(servo_pos_2, coincidence_2, 'o', label='February 24')
# plt.plot(relative_timing, coincidence, 'o')
# plt.plot(relative_timing, result.best_fit, '--k')
plt.xlim(x_range)
plt.ylim(y_range)
plt.title("HOM Coincidence Dip")
plt.xlabel("Timing Offset (mm)")
plt.ylabel("Coincidence Counts")
plt.legend()
plt.savefig(output_filename)

min_idx = np.argmin(df_1[bin_max])
print("Position of minimum: {} mm".format(servo_pos_1[min_idx]))
