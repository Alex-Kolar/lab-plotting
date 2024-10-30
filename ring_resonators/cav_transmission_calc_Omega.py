import numpy as np
import scipy
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import Model


DATA_ON = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
           "/New_mounted_device/10mK/09032024/SDS00002.csv")
FREQ_ON = (194811.486, 194819.973)  # unit: GHz

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
bbox = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor='black')
SAVE_FITS = False
SAVE_FIG = False

wavelength = 1539e-9  # units: m
c = 299792458  # units: m/s
frequency = c/wavelength

# from fitting of ions
# w_ion = 2.5e9  # unit: Hz
w_ion = 3.2e9
delta = 1.23e9  # unit: Hz

# from fitting of cavity (off-resonant)
w_cav = 4.241e9  # unit: Hz
cav_freq = 194815.726726  # unit: GHz
Q = 285018

k = frequency / Q  # units: 2*pi*Hz
gamma = 0


def W(w, delta, omega):
    return (-1j * (np.sqrt(np.pi*np.log(2))*np.square(omega)) / delta
            * np.exp(-np.square((w + .5j*gamma) / (delta / np.sqrt(np.log(2)))))
            * scipy.special.erfc(-1j*(w+.5j*gamma)/delta*np.sqrt(np.log(2))))


def T(w, a, b, phi, delta, w_ion, w_cav, omega):
    t = b*np.exp(1j*phi) + (-k/2)/(1j*(w-w_cav) - k/2 - 1j*W(w-w_ion, delta, omega))
    return a * t * np.conjugate(t)


def fit_func(x, amp, a, b, phi, omega):
    return amp*(1 - T(x, a=a, b=b, phi=phi, delta=delta, w_ion=w_ion, w_cav=w_cav, omega=omega))


# get cavity data
df_on = pd.read_csv(DATA_ON, header=10, skiprows=[11])
ramp = df_on['CH1'].astype(float).to_numpy()
transmission_on = df_on['CH2'].astype(float).to_numpy()

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
ramp = ramp[id_min:id_max]
transmission_on = transmission_on[id_min:id_max]

# convert time to frequency
freq_on = np.linspace(0, (FREQ_ON[1] - FREQ_ON[0]), id_max-id_min)  # unit: GHz
freq_on *= 1e9  # unit: Hz


# do fitting
model = Model(fit_func)
res = model.fit(transmission_on, x=freq_on,
                amp=0.7, a=0.6, b=0, phi=0, omega=100e6)
print(res.fit_report())


# plotting
fig, ax = plt.subplots()

freq_to_plot = (freq_on - w_cav) / 1e9  # convert to GHz
ax.plot(freq_to_plot, transmission_on,
        color='cornflowerblue', label='Data')
ax.plot(freq_to_plot, res.best_fit,
        ls='--', color='k', label='Fit')
ax.set_xlabel(f"Detuning from {cav_freq:.3f} (GHz)")
ax.set_ylabel("Cavity Reflection (A.U.)")
ax.set_title("On-resonant with Erbium Transition")
ax.set_xlim((-3, 3))
ax.legend(shadow=True)

omega = res.params['omega'].value / 1e6  # unit: MHz
omega_err = res.params['omega'].stderr / 1e6  # unit: MHz
label = rf"$\Omega$: {omega:.0f} $\pm$ {omega_err:.2f} MHz"
t = ax.text(0.95, 0.05, label,
            horizontalalignment='right', verticalalignment='bottom')
t.set_transform(ax.transAxes)

plt.tight_layout()
plt.show()
