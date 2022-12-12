"""
SAMWISE STAR

A (poor) model of stellar structure.

Specify the mass, [X], and chemical composition of a star
to calculate its structure from the differential equations
in terms of mass; the subsidiary relations for opacity, energy
generation rate, and the adiabatic exponent; and the associated
boundary conditions at the surface and center.

Uses shooting method with fourth-order
Runge-Kutta calculation and [XX].

Inputs
------
mass = mass of star (solar masses),
composition = hydrogen (H) and metal mass fractions (Z)
[X] = ???

Outputs
-------
A file called samwisestar_py.dat 
"""
import numpy as np


# constants
THIRD = 1 / 3
TWO_THIRD = 2 / 3
a = 7.56591e-15  # radiation constant (ergs cm^-3 K^-4)
c = 2.99792458e10  # speed of light (cm s^-1)
hydro = 1.673534e-24  # hydrogen mass (g)
grav = 6.67259e-8  # gravitational constant (cm^3 g^-1 s^-2)
k_b = 1.380658e-16  # Boltzmann constant (cm^2 g s^-2 K^-1)
N_A = 6.02214e23  # Avogadro's constant
sigma = 5.670374e-5  # Stephan-Boltzmann constant (erg cm^−2 s^−1 K^−4)
gradT_ad = 0.4  # adiabatic temperature gradient


# ============= DEPENDENCIES =============


def mmw(H, Z):
    """
    Compute mean molecular weight.
    
    Inputs
    ------
    H = hydrogen mass fraction,
    Z = metal mass fraction
    """
    HE = 1 - (H + Z)  # helium mass fraction
    mu = 1 / (2 * H + 0.75 * HE + 0.5 * Z)

    return mu


def density(P, T, mu):
    """
    Compute density from pressure, temperature, and mu.
    """
    P_rad = a * T**4 / 3
    P_gas = P - P_rad
    rho = (mu * hydro / k_b) * (P_gas / T)

    return rho


def energy_gen(H, Z, rho, T):
    """
    Compute energy generation rate from composition,
    density, and temperature.
    """
    HE = 1 - (H + Z)  # helium mass fraction
    T6 = T / 1e6
    e_pp = 2.4e4 * rho * H**2 * T6**(-TWO_THIRD) * np.exp((-3.38) / T6**THIRD)
    e_cno = 4.4e25 * rho * H * Z * T6**(-TWO_THIRD) * np.exp((-15.228) / T6**THIRD)
    e_3a = 5e8 * rho**2 * HE**3 / T6**3 * np.exp((-4.4) / T6)
    epsilon = e_pp + e_cno + e_3a

    return epsilon


def opacity(H, Z, rho, T):
    """
    Compute opacity from composition, density, and temperature.
    """
    k_bf = 4.34e25 * Z * (1 + H) * rho / T**3.5  # bound-free opacity
    k_ff = 3.68e22 * (1.0 - Z) * (1 + H) * rho / T**3.5  # free-free opacity
    k_e = 0.2 * (1 + H)  # electron scattering opacity
    k_h = 2.5e-31 * (Z / 0.02) * rho**0.5 * T**9
    kappa = 1 / ((1 / k_h) + (1 / (k_bf + k_ff + k_e)))

    return kappa


def adiab(P, T):
    """
    Compute adiabatic exponent from pressure and temperature.
    """
    P_rad = a * T**4 / 3
    P_gas = P - P_rad
    beta = P_gas / P
    gamma = (32 - (24*beta) - (3*beta**2)) / (24 - (18*beta) - (3*beta**2))
    return gamma


# ============= EQUATIONS OF STELLAR STRUCTURE =============


# MASS CONSERVATION
def dr_dm(radius, rho):
    """
    Derivative of radius with respect to mass.
    """
    return 1 / (4 * np.pi * radius**2 * rho)


# HYDROSTATIC EQUILIBRIUM
def dP_dm(mass, radius):
    """
    Derivative of pressure with respect to mass.
    """
    return (-1) * ((grav * mass) / (4 * np.pi * radius**4))


# ENERGY GENERATION
def dL_dm(epsilon):
    """
    Derivative of luminosity with respect to mass.
    """
    return epsilon


# ENERGY TRANSPORT
def dT_dm(H, Z, mass, radius, lum, press, temp):
    """
    Derivative of tempertature with respect to mass.
    """
    mu = mmw(H, Z)
    rho = (press, temp, mu)
    kappa = opacity(H, Z, rho, temp)
    gamma = adiab(press, temp)
    gamma_2 = (1 - (1 / gamma))

    # condition for convection
    if lum < ((16 * np.pi * a * c * grav / (3 * kappa)) * gamma_2 * temp**4 * mass / press):
        # radiation
        return (-1) * (3 * kappa * lum) / (64 * np.pi**2 * a * c * radius**4 * temp**3)
    else:
        # convection
        return (-1) * gamma_2 * grav * mass * temp / (4 * np.pi * radius**4 * press)


def deriv(ms, qs, H, Z):
    """
    Set of derivatives.

    Inputs
    ------
    ms = mass position for derivatives,
    qs = R, L, P, T for derivatives

    Outputs
    -------
    dR, dL, dP, dT
    """
    dq_dm = np.zeros(4)

    m = ms
    radius = qs[0]
    lum = qs[1]
    press = qs[2]
    temp = qs[3]

    mu = mmw(H, Z)
    rho = density(press, temp, mu)
    epsilon = energy_gen(H, Z, rho, temp)
    kappa = opacity(H, Z, rho, temp)

    dq_dm[0] = dr_dm(radius, rho)
    dq_dm[1] = dP_dm(m, radius)
    dq_dm[2] = dL_dm(epsilon)
    dq_dm[3] = dT_dm(H, Z, m, radius, lum, press, temp)

    return dq_dm


# ============= INNER BOUNDARIES =============


def center_radius(rho_c, m):
    """
    Central radius to start with.

    Inputs
    ------
    rho_c = central density,
    m = initial mass step
    """
    return ((3 * m) / (4 * np.pi * rho_c))**THIRD


def center_pressure(rho_c, P_c, m):
    """Central pressure boundary from central density,
    central pressure, and initial mass step.
    """
    delta_P = (3 * grav) / (8 * np.pi) * ((4*np.pi) / 3 * rho_c)**(4 / 3) * m**TWO_THIRD

    return P_c - delta_P


def center_temp(H, Z, rho_c, P_c, T_c, m, c=True):
    """
    Central temperature boundary from composition, central density,
    central pressure, central temperature, and initial mass step.

    Includes condition for convection or radiation.
    """
    if c:
        lnT = np.log(T_c) - np.pi / 6**THIRD * (gradT_ad * rho_c**(4/3)) / P_c * m**TWO_THIRD
        T = np.exp(lnT)
    else:
        kappa = opacity(H, Z, rho_c, T_c)  # correct rho?
        epsilon = energy_gen(H, Z, rho_c, T_c)  # correct rho and T?
        T4 = T_c**4 - 1 / (2 * a * c) * (3 / (4 * np.pi))**TWO_THIRD * kappa * epsilon * rho_c**(4/3) * m**TWO_THIRD
        T = T4**(1/4)
    return T


def center_star(P_c, T_c, mu, m, H, Z, c=False):
    """
    Central boundary conditions for the Samwise star.

    Inputs
    ------
    P_c = central pressure,
    T_c = central temperature,
    mu = mean molecular weight,
    m = initial mass step from center

    Outputs
    -------
    Equation of state values at point m = R, L, P, T
    """
    rho_c = density(P_c, T_c, mu)
    epsilon_c = energy_gen(H, Z, rho_c, T_c)
    R = center_radius(rho_c, m)
    L = m * epsilon_c
    P = center_pressure(rho_c, P_c, m)
    T = center_temp(H, Z, rho_c, P_c, T_c, m, c=c)

    return R, L, P, T


# ============= OUTER BOUNDARIES =============


def surface_temp(R, L):
    """
    Surface temperature from stellar radius and
    luminosity.
    """
    return (L / (4 * np.pi * R**2 * sigma))**(1/4)


def surface_star(R, L, mu):
    """
    Surface boundary conditions for the Samwise star.

    Inputs
    ------
    R = radius of star,
    L = luminosity of star,
    M = mass of star

    Outputs
    -------
    Equation of state values at point m = R, L, P, T
    """
    rho_s = 1e-5  # very small guess (g cm^-3)
    T = surface_temp(R, L)
    P = (THIRD * a * T**4) + (N_A * k_b * rho_s * T / mu)

    return R, L, P, T


# ============= SOLVER =============


def runge_center(H, Z, Ms):
    """???"""
    START = 1e-4 * Ms
    STOP = Ms / 2
    STEP = 1e-4 * Ms

    mu = mmw(H, Z)
    rho = density(P_c, T_c, mu)
    P = center_pressure(rho, P_c, m)
    T = center_temp(H, Z, rho, P_c, T_c, m, c=True)

    radii_0, lums_0, pressures_0, temps_0 = center_star(P, T, mu, m, H, Z, c=False)
    vector = np.array([radii_0, lums_0, pressures_0, temps_0], float)

    masses = np.arange(START, STOP, STEP)
    radii = []
    lums = []
    pressures = []
    temps = []

    for m in masses:
        radii.append(vector[0])
        lums.append(vector[1])
        pressures.append(vector[2])
        temps.append(vector[3])

        k1 = STEP * deriv(m, vector, H, Z)
        k2 = STEP * deriv(m + 0.5*STEP, vector + 0.5*k1, H, Z)
        k3 = STEP * deriv(m + 0.5*STEP, vector + 0.5*k2, H, Z)
        k4 = STEP * deriv(m + STEP, vector + k3, H, Z)

        vector += (k1 + 2*k2 + 2*k3 + k4) / 6

    return masses, vector


def runge_surface(H, Z, Ms):
