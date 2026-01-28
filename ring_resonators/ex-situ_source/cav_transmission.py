import numpy as np
import scipy


def W(w, delta, omega, gamma):
    return (-1j * np.sqrt(np.pi*np.log(2)) * np.square(omega)/delta) * \
        np.exp(-np.square((w+0.5j*gamma)/delta*np.sqrt(np.log(2)))) * \
        scipy.special.erfc(-1j*(w+0.5j*gamma)/delta*np.sqrt(np.log(2)))


def W_no_gamma(w, delta, omega):
    return (-1j * np.sqrt(np.pi*np.log(2)) * np.square(omega)/delta) * \
        np.exp(-np.square(w/delta*np.sqrt(np.log(2)))) * \
        scipy.special.erfc(-1j*w/delta*np.sqrt(np.log(2)))


# real transmission function
def T(w, w_cav, w_ions, kappa, delta, omega, gamma, a, b, phi):
    # complex field transmission function
    t = (b*np.exp(1j*phi)) + (-kappa/2) / (1j*(w-w_cav) - kappa/2 - 1j*W(w-w_ions, delta, omega, gamma))
    return a * (np.abs(t) ** 2)


def T_no_gamma(w, w_cav, w_ions, kappa, delta, omega, a, b, phi):
    # complex field transmission function
    t = (b*np.exp(1j*phi)) + (-kappa/2) / (1j*(w-w_cav) - kappa/2 - 1j*W_no_gamma(w-w_ions, delta, omega))
    return a * (np.abs(t) ** 2)


def R_no_gamma(w, w_cav, w_ions, kappa, delta, omega, a, b, phi):
    return a * (b**2 + 1) - T_no_gamma(w, w_cav, w_ions, kappa, delta, omega, a, b, phi)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # frequency parameters
    w = np.linspace(-5, 5, 1000)  # unit: GHz
    w_cav = 0
    w_ion = 0.5

    # cavity parameters
    kappa = 0.5

    # coupling parameters
    omega = 0.2  # unit: GHz
    delta = 0.3  # unit: GHz
    # gamma = 0

    # fitting parameters
    a = 1
    b = 0
    phi = 0

    # plotting
    transmission = T_no_gamma(w, w_cav, w_ion, kappa, delta, omega, a, b, phi)
    reflection = a - transmission
    transmission_no_ions = T_no_gamma(w, w_cav, w_ion, kappa, delta, 0, a, b, phi)
    reflection_no_ions = a - transmission_no_ions

    plt.plot(w, reflection_no_ions,
             label='Cavity Reflection')
    plt.plot(w, reflection,
             label='Cavity Reflection with Ions')
    plt.xlabel('Cavity Detuning (GHz)')
    plt.ylabel('Reflection')
    plt.legend()
    plt.tight_layout()
    plt.show()
