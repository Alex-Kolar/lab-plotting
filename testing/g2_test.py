import numpy as np
import matplotlib.pyplot as plt


HIGHEST_TAU = 10
# NUM_SAMPLES = 10 * HIGHEST_TAU
NUM_SAMPLES = int(1e6)

np.random.seed(0)

# generate realistic test points for 1 ms bin
data_dark = np.random.poisson(lam=0.002, size=NUM_SAMPLES)
data_ideal = np.random.binomial(n=1, p=0.01, size=NUM_SAMPLES)


def g2(data: np.ndarray):
    output = np.zeros(HIGHEST_TAU * 2 + 1)
    error = np.zeros(HIGHEST_TAU * 2 + 1)

    # common data
    avg_intensity = np.mean(data)
    avg_intensity_2 = avg_intensity ** 2
    error_intensity = np.std(data)
    error_intensity_2 = (2 * avg_intensity) * error_intensity
    error_intensity_ratio = error_intensity_2 / avg_intensity_2

    for tau in range(HIGHEST_TAU + 1):
        if tau == 0:
            zero_offset = data[:]
            coincidences = zero_offset * (zero_offset - 1)
        else:
            zero_offset = data[:-tau]
            tau_offset = data[tau:]
            coincidences = zero_offset * tau_offset

        # calculate g2
        avg_coincidences = np.mean(coincidences)
        g2_total = avg_coincidences / avg_intensity_2
        output[HIGHEST_TAU + tau] = g2_total
        output[HIGHEST_TAU - tau] = g2_total

        # calculate error
        error_coincidence = np.std(coincidences)
        error_coincidence_ratio = error_coincidence / avg_coincidences
        error_total = g2_total * np.sqrt((error_coincidence_ratio ** 2) +
                                         (error_intensity_ratio ** 2))
        error[HIGHEST_TAU + tau] = error_total
        error[HIGHEST_TAU - tau] = error_total

    return output, error


# calculate g2s
g2_dark, error_dark = g2(data_dark)
g2_ideal, error_ideal = g2(data_ideal)
g2_real, error_real = g2(data_dark + data_ideal)

# plotting
fig, ax = plt.subplots(2, 1)

x_coords = list(range(-HIGHEST_TAU, HIGHEST_TAU+1))

ax[0].bar(x_coords, g2_real,
          color='cornflowerblue', width=1, edgecolor='k', zorder=10)
# ax[0].errorbar(x_coords, g2_real, yerr=error_real,
#                ls='', color='k', zorder=10)

ax[1].bar(x_coords, g2_ideal,
          color='coral', width=1, edgecolor='k', zorder=10)

ax[0].grid(axis="y")
ax[1].grid(axis="y")
ax[0].set_xlim((-HIGHEST_TAU - 0.5, HIGHEST_TAU + 0.5))
ax[1].set_xlim((-HIGHEST_TAU - 0.5, HIGHEST_TAU + 0.5))

ax[0].set_title(r"All Counts")
ax[1].set_title(r"PL Only")
ax[1].set_xlabel(r"$\tau$ (ms)")
ax[0].set_ylabel(r"$g^{(2)}(\tau)$")
ax[1].set_ylabel(r"$g^{(2)}(\tau)$")

fig.tight_layout()
plt.show()
