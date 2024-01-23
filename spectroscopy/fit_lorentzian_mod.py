import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as colors
import numpy as np
from cmath import *
from math import *
from scipy.special import *
import csv
from lmfit.models import LinearModel, VoigtModel
import pandas as pd
import glob
import os
import matplotlib.cm as cm


plt.rc('font', family='Calibri', size=20)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


count = 0
curr_array = []
Matrix_ch1 = []
Matrix_freq = []
max_absorb_arr = []  # for setting max value of coloring on 2D plot
min_absorb_arr = []  # for setting min value of coloring on 2D plot

# how many lines to skip for header
linestart = 3
skipcount = 0

# g factor measurements
g_z1_parallel = 3.544
g_z1_perp = 7.085
g_y1_parallel = 4.51
g_y1_perp = 4.57

# frequency range scanned at full (unit GHz)
range_f = [10.616, 10.962, 10.520, 10.788, 10.749, 10.569]
max_f = [196045.536, 196045.795, 196045.574, 196045.832, 196045.590, 196045.560]
min_f = [196034.920, 196034.833, 196035.053, 196035.044, 196034.841, 196034.991]
F_RANGE_AVG = np.mean(range_f)
F_MAX_AVG = np.mean(max_f)
F_MIN_AVG = np.mean(min_f)

# max current (A)
CURR_MAX = 10

# path to csv for wide measurement
PATH_FULL = '.\\fullscale\\full00.csv'

# acquire maximum and minimum values for frequency modulation
full_df = pd.read_csv(PATH_FULL)
full_df.drop(0, inplace=True)  # remove line for units
full_df = full_df.astype(float)
voltages = full_df['CH4']
volt_min = np.min(voltages)
volt_range = np.max(voltages) - volt_min


# Get rubidium data
PATH_RB = ".\\rubidium\\scan1530 zoomed out.csv"
df_rb = pd.read_csv(PATH_RB, header=0, names=['TTL','L1','L2'], usecols=[1,2,3], skiprows=[1])

# rubidium frequency data
centerfreq_rb = 196037.1371  #GHz
maxfreq_rb = 196037.2781
minfreq_rb = 196036.9833

# process rubidium data
start = 535
stop = 1040
ttl = df_rb['TTL'][start:stop]
hi = np.max(ttl)

y = df_rb['L1']
n = y.shape[0]
avg_emit_rb = np.zeros(n)  # moving window average of emission

for i in range(n):
    if i == 0 or i == n-1:
        continue
    else:
        avg_emit_rb[i] = (y[i-1] + y[i] + y[i+1])/3

i_00=0
i_01=0
i_max=0
i_min=0
for i in range(start, stop-1):
    if ttl[i] < hi/2 and ttl[i+1] > hi/2:
        i_00 = i
    elif ttl[i] > hi/2 and ttl[i+1] < hi/2:
        i_01 = i
        
i_max = int((i_00 + i_01)/2)
ghz_rb = np.linspace(centerfreq_rb, maxfreq_rb, num=i_max-i_00)
ghz_rb -= F_MIN_AVG
avg_emit_rb = avg_emit_rb[i_00:i_max]
avg_emit_rb /= (3*np.max(avg_emit_rb))


for filename in sorted(glob.glob('*.csv'), key=os.path.getmtime):
    try:
        count += 1
        with open(filename, 'r') as csvfile:

            timearray = []
            ch1array = []
            ch4array = []

            # bfield should be filename
            current = float(os.path.splitext(filename)[0]) / 10
            curr_array.append(current)

            csvreader = csv.reader(csvfile, delimiter=",")
            linenumber = 0
            for row in csvreader:
                linenumber += 1

                # skip headers
                if linenumber < linestart:
                    continue

                time=float(row[0])
                ch1=float(row[1])
                ch4=float(row[2])

                timearray.append(time)
                ch1array.append(ch1)
                ch4array.append(ch4)

            timearray=np.array(timearray)
            ch1array=np.array(ch1array)
            ch4array=np.array(ch4array)
    
        # get maximum value and id for csv
        maxidch4 = np.argmax(ch4array, axis=0)
        maxch4 = np.max(ch4array, axis=0)
        minidch4 = np.argmin(ch4array, axis=0)
        minch4 = np.min(ch4array, axis=0)
        
        ch1array = ch1array[minidch4:maxidch4]
        # ch1array = ch1array/np.max(ch1array)  # normalize

        time_arr_start = (minch4 - volt_min) * (F_RANGE_AVG/volt_range)
        time_arr_stop = (maxch4 - volt_min) * (F_RANGE_AVG/volt_range)
        timearray = np.linspace(time_arr_start, time_arr_stop, num=len(ch1array))

        # # plotting
        # offset = 0.3
        # color_val = (bfield/B_MAX) * (1 - offset) + offset
        # color = cm.Blues(color_val)
        # # plt.plot(timearray, (ch1array + bfield/20)*(10/6), color=color)
        # plt.plot(timearray, ch1array + bfield, color=color)

        Matrix_ch1.append(ch1array)
        Matrix_freq.append(timearray)
        max_absorb_arr.append(np.max(ch1array))
        min_absorb_arr.append(np.min(ch1array))

    except FileNotFoundError:
        print('File not found:', filename)
        break

# get max and min values for coloring
max_absorb = np.max(max_absorb_arr)
min_absorb = np.min(min_absorb_arr)
# convert to numpy array
curr_array = np.array(curr_array)


# do fitting
sigmas = []
centers = []
fits = []
for freqs, ch1 in zip(Matrix_freq, Matrix_ch1):
    ch1 = ch1/np.max(ch1)
    # parameter guesses
    center_guess = freqs[np.argmin(ch1)]

    # idx_range = np.where(np.abs(freqs - center_guess) < full_width)
    # ch1_trunc = ch1[idx_range]
    # freq_trunc = freqs[idx_range]

    peak = VoigtModel()
    background = LinearModel()
    model = peak + background
    res = model.fit(1 - ch1, x=freqs, center=center_guess)
    sigmas.append(res.params['sigma'].value)
    centers.append(res.params['center'].value)
    fits.append(res.best_fit)
    # print(res.fit_report())

    # uncomment these lines to plot each individual fitting
    # plt.plot(freqs, 1 - ch1)
    # plt.plot(freqs, res.best_fit)
    # plt.show()

sigmas = np.array(sigmas)
centers = np.array(centers)

# fitting parameters for finding slope, etc.
min_curr = 4.5
idx_to_keep = np.where(curr_array >= min_curr)
Barray = curr_array[idx_to_keep]
sigmas = sigmas[idx_to_keep]
centers = centers[idx_to_keep]

# fit centers to get B field
g_linear = LinearModel()
res_linear = g_linear.fit(centers, x=Barray)
# print(res_linear.fit_report())
slope = -res_linear.params['slope'].value  # units: GHz/A

# calculations
h = 4.136e-6  # unit: eV/GHz
mu_B = 5.788e-5  # unit: eV/T
g = (g_z1_parallel + g_y1_parallel) / 2
slope_B = (slope * h) / (mu_B * g)  # units: T/A
print("slope_B =", slope_B)
b_array = curr_array * slope_B
b_array *= 1e3  # convert to mT

# plotting change in sigma and center
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

color = 'tab:blue'
ax1.plot(b_array[idx_to_keep], sigmas, 'o', color=color)
ax1.set_ylabel("Linewidth (GHz)", color=color)
ax1.tick_params(axis='y', labelcolor=color)

color = 'tab:orange'
ax2.plot(b_array[idx_to_keep], centers, 'o', color=color)
ax2.set_ylabel("Center (GHz)", color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.plot(b_array[idx_to_keep], res_linear.best_fit, color='k')

ax1.set_xlabel('Magnetic Field (mT)')
ax1.set_title("Fitted Parameters, Polarization 1.5")
plt.tight_layout()
plt.show()


# plotting of all lines
OFFSET = 0.3
lines = []
for freq, ch1, current in zip(Matrix_freq, Matrix_ch1, curr_array):
    ch1_plotting = (ch1/np.max(ch1)) + current
    line = np.column_stack((freq, ch1_plotting))
    lines.append(line)

cmap = truncate_colormap(cm.Greens, OFFSET, 1)
line_coll = LineCollection(lines, cmap=cmap)
line_coll.set_array(b_array)

fig, ax = plt.subplots()
ax.add_collection(line_coll, autolim=True)
ax.autoscale_view()

# add rubidium spectrum
CURRENT_SIM_RB = 10
plt.plot(ghz_rb, avg_emit_rb + CURRENT_SIM_RB, color='maroon')

axcb = fig.colorbar(line_coll, ax=ax)
axcb.set_label("Magnetic Field (mT)")
ax.tick_params(left=False, labelleft=False)
ax.set_xlabel("Detuning from {:.3f} (GHz)".format(F_MIN_AVG))
ax.set_ylabel("Transmission (A.U.)")
ax.set_title("Polarization 1.5")

plt.tight_layout()
plt.show()


# plotting of only lines near 0 magnetic field
LINES_TO_INCLUDE = 3

line_coll = LineCollection(lines[-LINES_TO_INCLUDE:], cmap=cmap)
line_coll.set_array(b_array[-LINES_TO_INCLUDE:])

fig, ax = plt.subplots()
ax.add_collection(line_coll, autolim=True)
ax.autoscale_view()
axcb = fig.colorbar(line_coll, ax=ax)
axcb.set_label("Magnetic Field (mT)")
ax.tick_params(left=False, labelleft=False)
ax.set_xlabel("Detuning from {:.3f} (GHz)".format(F_MIN_AVG))
ax.set_ylabel("Transmission (A.U.)")
ax.set_title("Polarization 1.5")

plt.tight_layout()
plt.show()


# plot of rubidium and single line
CURR_TO_PLOT = 9
x_range = (1.8, 3.2)  # for 9 A
# x_range = (1.3, 2.7)  # for 9.5 A
y_range_er = (0.75, 0.98)

absorb_idx = np.argmin(np.abs(curr_array - CURR_TO_PLOT))
ch1 = Matrix_ch1[absorb_idx]/np.max(Matrix_ch1[absorb_idx])
b_field = b_array[absorb_idx]
print("b_field =", b_field, "(mT)")

# get color for plotting of single line
color_frac = (CURR_TO_PLOT - np.min(curr_array)) / (np.max(curr_array) - np.min(curr_array))
color = cmap(color_frac)
print(color)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# color = 'olivedrab'
er_line = ax1.plot(Matrix_freq[absorb_idx], ch1, color=color, label='Erbium')
fit_line = ax1.plot(Matrix_freq[absorb_idx], 1-fits[absorb_idx], ls='--', color='k', label='Erbium Fit')
ax1.set_ylabel("1530 nm laser transmission (A.U.)", color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(y_range_er)

color = 'maroon'
rb_line = ax2.plot(ghz_rb, avg_emit_rb + CURRENT_SIM_RB, color=color, label='Rubidium')
ax2.set_ylabel("780 nm laser transmission (A.U.)", color=color)
ax2.tick_params(axis='y', labelcolor=color)

ax1.set_xlim(x_range)
ax1.set_xlabel("Detuning from {:.3f} (GHz)".format(F_MIN_AVG))

lines = er_line + fit_line + rb_line
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower left')

plt.tight_layout()
plt.show()


# 2D plot of absorption
fig, ax = plt.subplots()
ax.set_facecolor('k')

# define y grid
Y = np.zeros(b_array.shape[0] + 1)
Y[1:-1] = b_array[0:-1] + np.diff(b_array)/2
Y[0] = b_array[0] - np.diff(b_array)[0]/2
Y[-1] = b_array[-1] + np.diff(b_array)[-1]/2
for i, (freq, ch1) in enumerate(zip(Matrix_freq, Matrix_ch1)):
    X = np.zeros(freq.shape[0] + 1)
    X[1:-1] = freq[0:-1] + np.diff(freq)/2
    X[0] = freq[0] - np.diff(freq)[0]/2
    X[-1] = freq[-1] + np.diff(freq)[-1]/2
    im = ax.pcolormesh(X, Y[i:i+2], [ch1], cmap='Blues', vmin=min_absorb, vmax=max_absorb)

axcb = fig.colorbar(im, ax=ax)
axcb.set_label("Absorption (A.U.)")
ax.set_xlabel("Detuning from {:.3f} (GHz)".format(F_MIN_AVG))
ax.set_ylabel("Magnetic Field (mT)")
ax.set_title("Polarization 1.5")

plt.tight_layout()
plt.show()


# fig,ax=plt.subplots()

# freqarray=np.array(freqarray)
# Matrix=np.array(Matrix)


# MatrixB=np.asarray(Matrix.reshape(count,len(freqarray)))

# # MatrixB=MatrixB.transpose()
# # MatrixB=np.flip(MatrixB,0)
# starvalue=freqarray.min()
# endvalue=freqarray.max()

# Barray=np.array(Barray)
# im = ax.imshow(MatrixB, extent=(Barray.min(), Barray.max(),starvalue,endvalue), interpolation='none', cmap=cm.jet, aspect="auto")
# cbar = ax.figure.colorbar(im, ax=ax)
# # ax = seaborn.heatmap(Matrix,xticklabels=x_axis_labels, yticklabels=y_axis_labels)
# # cbar = ax.figure.colorbar(im, ax=ax)
# # ax.set_title('Freq (GHz)',fontsize=20)
# plt.xlabel("B field (mT)",fontsize=20)
# plt.ylabel("Freq (GHz)",fontsize=20)
# # plt.xlim(30,140)
# plt.tight_layout()
# ax = plt.gca()
# ax.tick_params(labelsize=20)

# # plt.show()