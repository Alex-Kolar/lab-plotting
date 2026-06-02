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
    return a * np.abs(b*np.exp(1j*phi) + 1)**2 - T_no_gamma(w, w_cav, w_ions, kappa, delta, omega, a, b, phi)


def R_mod(w, w_cav, w_ions, kappa, kappa_in, coupling, inhomog, a):
    ion_term = (coupling ** 2) / (w - w_ions + 1j*inhomog/2)
    t = 1 - (1j*kappa_in)/(w - w_cav + 1j*kappa/2 - ion_term)
    return a * (np.abs(t) ** 2)


def R_mod_lmfit(x, w_cav, w_ions, kappa, kappa_in, coupling, inhomog, a):
    ion_term = (coupling ** 2) / (x - w_ions + 1j * inhomog / 2)
    t = 1 - (1j*kappa_in)/(x - w_cav + 1j * kappa / 2 - ion_term)
    return a * (np.abs(t) ** 2)


"""Adding different versions for different ion distributions"""
def W_lorentzian(x, w_ions, coupling, inhomog):
    return (coupling ** 2) / (x - w_ions + 1j*inhomog/2)

def W_gaussian(x, w_ions, coupling, inhomog):
    log2 = np.log(2)
    interior_term = (x - w_ions) / inhomog * np.sqrt(log2)
    return (-1j * np.sqrt(np.pi * log2) * np.square(coupling) / inhomog) * \
        np.exp(-np.square(interior_term)) * \
        scipy.special.erfc(-1j * interior_term)

def R_mod_lmfit_lorentzian(x, w_cav, w_ions, kappa, kappa_in, coupling, inhomog, a):
    ion_term = W_lorentzian(x, w_ions, coupling, inhomog)
    t = 1 - (1j*kappa_in)/(x - w_cav + 1j * kappa / 2 - ion_term)
    return a * (np.abs(t) ** 2)

def R_mod_lmfit_gaussian(x, w_cav, w_ions, kappa, kappa_in, coupling, inhomog, a):
    ion_term = W_gaussian(x, w_ions, coupling, inhomog)
    t = 1 - (1j*kappa_in)/(x - w_cav + 1j * kappa / 2 - ion_term)
    return a * (np.abs(t) ** 2)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # frequency parameters
    w = np.linspace(-2, 2, 1000)  # unit: GHz
    w_cav = 0
    w_ion = 0

    # cavity parameters
    kappa = 0.396  # unit: GHz
    kappa_in = kappa / 2

    # coupling parameters
    # coupling = 0.266  # unit: GHz
    coupling = 0.2
    inhomogeneous = 0.384  # unit: GHz
    # gamma = 0

    # fitting parameters
    a = 1
    b = 1
    phi = 0

    # plotting
    # transmission = T_no_gamma(w, w_cav, w_ion, kappa, inhomogeneous, coupling, a, b, phi)
    # reflection = R_no_gamma(w, w_cav, w_ion, kappa, inhomogeneous, coupling, a, b, phi)
    # transmission_no_ions = T_no_gamma(w, w_cav, w_ion, kappa, inhomogeneous, 0, a, b, phi)
    # reflection_no_ions = R_no_gamma(w, w_cav, w_ion, kappa, inhomogeneous, 0, a, b, phi)
    reflection = R_mod_lmfit_gaussian(w, w_cav, w_ion, kappa, kappa_in, coupling, inhomogeneous, a)
    reflection_no_ions = R_mod_lmfit_lorentzian(w, w_cav, w_ion, kappa, kappa_in, 0, inhomogeneous, a)

    plt.plot(w, reflection_no_ions,
             label='Cavity Reflection')
    plt.plot(w, reflection,
             label='Cavity Reflection with Ions')
    plt.xlabel('Cavity Detuning (GHz)')
    plt.ylabel('Reflection')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # plt.plot(w, transmission_no_ions,
    #          label='Cavity Reflection')
    # plt.plot(w, transmission,
    #          label='Cavity Reflection with Ions')
    # plt.xlabel('Cavity Detuning (GHz)')
    # plt.ylabel('Reflection')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
