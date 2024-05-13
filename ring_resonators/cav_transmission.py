# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 10:02:06 2024

@author: ianac
"""

import numpy as np
import math
import scipy

import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt


w_cav = 0
ws_ion = [0, 1]

Q = 1000000
wavelength = 1550e-9  # units: m
c = 299792458  # units: m/s
frequency = c/wavelength

omega = .537
k = frequency*2*np.pi/Q*1e-9  #2.43*2*np.pi
deltas = [0.2, 200]
gamma = 0

a = 1
b = 0


def W(w, delta):
    return -1j*np.sqrt(np.pi*np.log(2))*np.square(omega)/delta*np.exp(-np.square((w+.5j*gamma)/delta*np.sqrt(np.log(2))))*scipy.special.erfc(-1j*(w+.5j*gamma)/delta*np.sqrt(np.log(2)))


def T(w, delta, w_ion):
    t = b+(-k/2)/(1j*(w-w_cav)-k/2-1j*W(w-w_ion, delta))
    return a*t*np.conjugate(t)


def T_no_ion(w):
    t = b+(-k/2)/(1j*(w-w_cav)-k/2)
    return a*t*np.conjugate(t)


w_ = np.linspace(-10, 10, 1000)
transmit_small = np.zeros_like(w_)  # small delta
transmit_large = np.zeros_like(w_)  # large delta
transmit = np.zeros_like(w_)

for i in range(len(w_)):
    transmit_small[i] = T(w_[i], deltas[0], ws_ion[0])
    transmit_large[i] = T(w_[i], deltas[1], ws_ion[0])
    transmit[i] = T_no_ion(w_[i])

# plot reflection
plt.plot(w_, transmit,
         color='k', label="No Ion")
plt.plot(w_, transmit_small,
         color='cornflowerblue', ls='--',
         label=r"$\Delta$ = {} GHz".format(deltas[0]))
plt.plot(w_, transmit_large,
         color='coral', ls='--',
         label=r"$\Delta$ = {} GHz".format(deltas[1]))

plt.grid()
plt.legend()
plt.title("Changing Inhomogeneous Broadening")
plt.xlabel("Detuning (GHz)")
plt.ylabel("Cavity Transmission")

plt.tight_layout()
plt.show()
