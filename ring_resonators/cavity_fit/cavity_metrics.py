import numpy as np


# fitting function for coincidences
def g_2_no_delta(x, x0, amplitude, kappa, g):
    x = np.abs(x - x0)
    exp_term = g*np.sinh(g*x) + (kappa/2)*np.cosh(g*x)
    g_2 = 1 + (np.exp(-kappa * x) / (g ** 2)) * (np.abs(exp_term) ** 2)
    return amplitude * g_2


def calculate_gamma(centers, qs, contrasts, a,
                    L=(np.pi*220e-6), n_eff=2.18):
    """Calculate nonlinear coefficient gamma."""
    # speed of light
    c = 3e8

    enhance_1, enhance_2 = calculate_enhancement(centers, qs, contrasts, L, n_eff)

    numerator = 2 * n_eff * a
    denominator_1 = (enhance_1 ** 3) * c * L
    denominator_2 = (enhance_2 ** 3) * c * L

    gamma_1 = np.sqrt(numerator / denominator_1)
    gamma_2 = np.sqrt(numerator / denominator_2)

    return gamma_1, gamma_2


def calculate_enhancement(centers, qs, contrasts,
                          L=(np.pi*220e-6), n_eff=2.18):
    """Calculate power enhancement factor |F_0\^2.

    Args:
        centers (np.ndarray): frequency centers of resonances, in THz.
        qs (np.ndarray): Q factors for resonances.
        contrasts (np.ndarray): contrast of resonances.
    """
    # speed of light
    c = 3e8

    # calculate |kappa|^2 and |t|^2 using Q
    lambda_0 = c / (centers * 1e12)  # unit: m
    kappa_squared = (np.pi * L * n_eff) / (lambda_0 * qs)
    t_squared = 1 - kappa_squared
    t = np.sqrt(t_squared)

    # calculate alpha (two possible values)
    norm_trans = 1 - contrasts
    alpha_1 = (norm_trans + t) / (norm_trans * t + 1)
    alpha_2 = (norm_trans - t) / (norm_trans * t - 1)

    # calculate normalized power (field enhancement squared) on device
    norm_power_1 = ((alpha_1 ** 2) * (1 - t_squared)) / ((1 - alpha_1 * t) ** 2)
    norm_power_2 = ((alpha_2 ** 2) * (1 - t_squared)) / ((1 - alpha_2 * t) ** 2)

    return norm_power_1, norm_power_2


def calculate_rates(centers, qs, contrasts,
                    L=(np.pi*220e-6), n_eff=2.18, gamma=1.66e-5, power=1):
    """Calculate the expected pair rate of generated photons for SiC ring resonators.

    First calculates the field enhancement factor F_0, based on measured Q and contrast.
    Note that two solutions are possible, corresponding to the under- and over-coupled cavity cases.
    Both values are returned by the function.

    Args:
        centers (np.ndarray): frequency centers of resonances, in THz.
        qs (np.ndarray): Q factors for resonances.
        contrasts (np.ndarray): contrast of resonances.
        L (float): cavity circumference, in m (default pi * 220e-6).
        n_eff (float): effective cavity index (default 2.18).
            Default is calculated based on MATLAB simulation.
        gamma (float): nonlinearity coefficient (default 884.012153).
            Default is calculated based on fit of power scan.
        power (float): pump power, default 1 (in mW).

    Returns:
        Tuple[np.ndarray, np.ndarray]: pair rates, with unit pairs/s
    """
    # speed of light
    c = 3e8

    # calculate |kappa|^2 and |t|^2 using Q
    lambda_0 = c / (centers * 1e12)  # unit: m
    kappa_squared = (np.pi * L * n_eff) / (lambda_0 * qs)
    t_squared = 1 - kappa_squared
    t = np.sqrt(t_squared)

    # calculate alpha (two possible values)
    norm_trans = 1 - contrasts
    alpha_1 = (norm_trans + t) / (norm_trans*t + 1)
    alpha_2 = (norm_trans - t) / (norm_trans*t - 1)

    # calculate normalized power (field enhancement squared) on device
    norm_power_1 = ((alpha_1 ** 2) * (1 - t_squared)) / ((1 - alpha_1 * t) ** 2)
    norm_power_2 = ((alpha_2 ** 2) * (1 - t_squared)) / ((1 - alpha_2 * t) ** 2)

    # calculate photons per second generated on sideband
    n_photons_1 = ((gamma * power) ** 2) * (norm_power_1 ** 3) * ((c * L) / (2 * n_eff))
    n_photons_2 = ((gamma * power) ** 2) * (norm_power_2 ** 3) * ((c * L) / (2 * n_eff))

    return n_photons_1, n_photons_2
