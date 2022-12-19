"""
SAMWISE STAR

A (poor) model of stellar structure.

Specify the mass, luminosity, temperature, and chemical
composition of a star to calculate its structure from the
differential equations in terms of mass, and the subsidiary
relations for opacity, energy generation rate, and density.

Uses a Runge-Kutta algorithm to solve the equations from
the surface of the star to its center.

Input
------
mass = mass of star (solar mass),
luminosity = luminosity of the star (solar luminosity),
temperature = temperature of the star (K),
composition = hydrogen and metal mass fractions

Output
-------
A file containing the model called samwise_star.dat
"""
import numpy as np


# ============= CONSTANTS (cgs) =============

rad_dens = 7.565e-15  # radiation density constant (erg cm^-3 K^-4)
light = 2.99792458e10  # speed of light (cm s^-1)
h_mass = 1.6735e-24  # hydrogen atom mass (g)
boltz = 1.3807e-16  # Boltzmann constant (cm^2 g s^-2 K^-1)
grav = 6.6743e-8  # gravitational constant (cm^3 g^-1 s^-2)
stef_boltz = 5.6704e-5  # Stefan-Boltzmann (erg cm^−2 s^−1 K^−4)
gamma = 5 / 3  # adiabatic exponent for monatomic gas
N_A = 6.02214076e23  # Avogadro constant (mol^-1)
M_sun = 1.989e33  # solar mass (g)
L_sun = 3.839e33  # solar luminosity (erg s^-1)
R_sun = 6.955e10  # solar radius (cm)

# ============= DEPENDENCIES =============


def mmw(hydro, metal):
    """
    Compute mean molecular weight.

    Input
    ------
    hydro = hydrogen mass fraction,
    metal = metal mass fraction
    """
    helium = 1 - (hydro + metal)  # helium mass fraction
    mean_mw = 1 / (2 * hydro + 0.75 * helium + 0.5 * metal)

    return mean_mw


def density(press, temp, mean_mol):
    """
    Compute density.

    Input
    ------
    press = pressure,
    temp = temperature,
    mean_mol = mean molecular weight
    """
    press_rad = rad_dens * temp**4 / 3  # radiation pressure
    press_gas = press - press_rad  # gas pressure
    dens = (mean_mol * h_mass / boltz) * (press_gas / temp)

    return dens


def energy_gen(hydro, metal, dens, temp):
    """
    Compute energy generation rate.

    Input
    ------
    hydro = hydrogen mass fraction,
    metal = metal mass fraction,
    dens = density,
    temp = temperature
    """
    T6 = temp / 1e6
    e_pp = 2.4e4 * dens * hydro**2 * T6**(-2/3) * np.exp((-3.38) / T6**(1/3))  # proton-proton chain
    e_cno = 4.4e25 * dens * hydro * metal * T6**(-2/3) * np.exp((-15.228) / T6**(1/3))  # CNO cycle
    e_gen = e_pp + e_cno

    return e_gen


def opacity(hydro, metal, dens, temp):
    """
    Compute opacity.

    Input
    ------
    hydro = hydrogen mass fraction,
    metal = metal mass fraction,
    dens = density,
    temp = temperature
    """
    gaunt = 0.01  # manually set gaunt factor
    k_bf = 4.34e25 / gaunt * metal * (1 + hydro) * dens / temp**3.5  # bound-free opacity
    k_ff = 3.68e22 * (1 - metal) * (1 + hydro) * dens / temp**3.5  # free-free opacity
    k_e = 0.2 * (1 + hydro)  # electron scattering opacity
    opac = k_bf + k_ff + k_e

    return opac


# ============= EQUATIONS OF STELLAR STRUCTURE =============


# MASS CONSERVATION
def dr_dm(rad, dens):
    """
    Derivative of radius with respect to mass.

    Input
    -----
    rad = radius,
    dens = density
    """
    return 1 / (4 * np.pi * rad**2 * dens)


# HYDROSTATIC EQUILIBRIUM
def dP_dm(mas, rad):
    """
    Derivative of pressure with respect to mass.

    Input
    -----
    mas = mass,
    rad = radius
    """
    return (-1) * ((grav * mas) / (4 * np.pi * rad**4))


# ENERGY GENERATION
def dL_dm(eps):
    """
    Derivative of luminosity with respect to mass.

    Input
    -----
    eps = epsilon
    """
    return eps


# ENERGY TRANSPORT
def dT_dm(hydro, metal, mas, rad, lum, press, temp, trans):
    """
    Derivative of tempertature with respect to mass.

    Input
    -----
    hydro = hydrogen mass fraction,
    metal = metal mass fraction,
    mas = mass,
    rad = radius,
    lum = luminosity,
    press = pressure,
    temp = temperature,
    trans = energy transport method
    """
    mu = mmw(hydro, metal)
    rho = density(press, temp, mu)
    kappa = opacity(hydro, metal, rho, temp)
    gamma2 = (1 - (1 / gamma))

    if trans == 0:
        # radiation
        return ((-3) * kappa * lum) / (64 * np.pi**2 * rad_dens * light * rad**4 * temp**3)
    else:
        # convection
        return (-gamma2) * grav * mas * temp / (4 * np.pi * rad**4 * press)


# ============= ALGORITHM =============


def derivatives(hydro, metal, mas, values, trans):
    """
    Set of derivatives.

    Input
    ------
    hydro = hydrogen mass fraction,
    metal = metal mass fraction,
    mas = mass,
    values = R, L, P, T for derivatives,
    trans = energy transport method

    Output
    -------
    df_dm = dR, dL, dP, dT
    """
    df_dm = np.zeros(4)

    rad_m = values[0]
    press_m = values[1]
    lum_m = values[2]
    temp_m = values[3]

    mu = mmw(hydro, metal)
    rho = density(press_m, temp_m, mu)
    epsilon = energy_gen(hydro, metal, rho, temp_m)

    df_dm[0] = dr_dm(rad_m, rho)
    df_dm[1] = dP_dm(mas, rad_m)
    df_dm[2] = dL_dm(epsilon)
    df_dm[3] = dT_dm(hydro, metal, mas, rad_m, lum_m, press_m, temp_m, trans)

    return df_dm


def runge_kutta(hydro, metal, values, df_dm, mas, delta_m, trans):
    """
    Runge-Kutta step to compute values in
    mass shells of star.

    Input
    -----
    hydro = hydrogen mass fraction,
    metal = metal mass fraction,
    values = R, L, P, T for derivatives,
    df_dm = derivate values,
    mas = mass,
    delta_m = change in mass,
    trans = energy transport method
    """
    temp_values = np.zeros(4)
    results = np.zeros(4)

    dm_2 = delta_m / 2
    dm_6 = delta_m / 6
    mas_2 = mas + dm_2
    mas_next = mas + delta_m

    for i in range(0, 4):
        temp_values[i] = values[i] + dm_2 * df_dm[i]

    df1 = derivatives(hydro, metal, mas_2, temp_values, trans)

    for i in range(0, 4):
        temp_values[i] = values[i] + dm_2 * df1[i]

    df2 = derivatives(hydro, metal, mas_2, temp_values, trans)

    for i in range(0, 4):
        temp_values[i] = values[i] + delta_m * df2[i]

    df3 = derivatives(hydro, metal, mas_next, temp_values, trans)

    # next mass shell
    for i in range(0, 4):
        results[i] = values[i] + dm_6 * (df_dm[i] + 2 * df1[i] + 2 * df2[i] + df3[i])

    return results


# ============= MAIN ROUTINE =============


def samwise_star(M_star, L_star, T_star, hydro, metal):
    """
    Main routine for calculating stellar structure.

    step_flag = set step size
            = 0 (initial surface step size of M_star / 1000)
            = 1 (interior step size of M_star / 100)
            = 2 (core step size of M_star / 5000)
    error_flag = final model condition flag
        = -1 (number of zones exceeded; also the initial value)
        =  0 (good model)
        =  1 (core density was extreme)
        =  2 (core luminosity was extreme)
        =  3 (core temperature was too low)
        =  4 (mass became negative before center was reached)
        =  5 (luminosity became negative before center was reached)

    Input
    ------
    M_star = mass of star,
    L_star = luminosity of star (solar mass),
    T_star = temperature of star (solar luminosity),
    hydro = mass fraction of hydrogen,
    metal = mass fractioN of metals

    Output
    ------
    A file containing the model called samwise_star.dat
    """
    # routine parameters
    shells = 999  # max shells allowed in star
    dlPlim = 99.9  # output limit to avoid format field overflows
    step_flag = 0  # see docstring
    error_flag = -1  # see docstring

    # initializing structure equations
    values = np.zeros(4, float)
    derivs = np.zeros(4, float)
    results = np.zeros(4, float)

    # initializing variables
    m = np.zeros(shells, float)
    Rad_m = np.zeros(shells, float)
    Press_m = np.zeros(shells, float)
    Lum_m = np.zeros(shells, float)
    Temp_m = np.zeros(shells, float)
    rho = np.zeros(shells, float)
    kappa = np.zeros(shells, float)
    epsilon = np.zeros(shells, float)
    dlPdlT = np.zeros(shells, float)  # dlPdlT = dlnP/dlnT

    # properties of star at surface
    helium = 1 - (hydro + metal)
    mu = mmw(hydro, metal)
    gamma_index = gamma / (gamma - 1)

    rho_surface = 1e-5  # very small guess
    M_surface = M_star * M_sun
    L_surface = L_star * L_sun
    P_surface = (rad_dens * T_star**4 / 3) + ((N_A * boltz / mu) * rho_surface * T_star)
    T_surface = T_star
    R_surface = (L_surface / (4 * np.pi * stef_boltz * T_star**4))**0.5
    R_star = R_surface / R_sun

    # starting values
    delta_m = -M_surface / 1000  # mass integration step (negative for inward)
    start = 0
    m[start] = M_surface
    Rad_m[start] = R_surface
    Press_m[start] = P_surface
    Lum_m[start] = L_surface
    Temp_m[start] = T_surface
    rho[start] = density(Press_m[start], Temp_m[start], mu)
    kappa[start] = opacity(hydro, metal, rho[start], Temp_m[start])
    epsilon[start] = energy_gen(hydro, metal, rho[start], Temp_m[start])
    dlPdlT[start] = 4.25  # arbitrary starting value
    trans = 1  # energy transport assumption

    # loop
    for i in range(1, shells):
        last = i - 1
        values[0] = Rad_m[last]
        values[1] = Press_m[last]
        values[2] = Lum_m[last]
        values[3] = Temp_m[last]

        for k in range(0, 4):
            df_dm = derivatives(hydro, metal, m[last], values, trans)
            derivs[k] = df_dm[k]

        results = runge_kutta(hydro, metal, values, derivs, m[last], delta_m, trans)

        m[i] = m[last] + delta_m
        Rad_m[i] = results[0]
        Press_m[i] = results[1]
        Lum_m[i] = results[2]
        Temp_m[i] = results[3]

        # calculate dependencies for this shell
        rho[i] = density(Press_m[i], Temp_m[i], mu)
        kappa[i] = opacity(hydro, metal, rho[i], Temp_m[i])
        epsilon[i] = energy_gen(hydro, metal, rho[i], Temp_m[i])

        # determine energy transport in next shell
        dlPdlT[i] = np.log(Press_m[i] / Press_m[last]) / np.log(Temp_m[i] / Temp_m[last])
        if dlPdlT[i] < gamma_index:
            trans = 1
        else:
            trans = 0

    # check whether core has been reached, set error_flag and
    # estimate the central conditions
        if (Rad_m[i] <= 0) and (Lum_m[i] >= (0.1*L_surface)) or (m[i] >= (0.01*M_surface)):
            error_flag = 6

        elif Lum_m[i] <= 0:
            error_flag = 5
            rho_core = m[i]/(4/3*np.pi*Rad_m[i]**3)
            if m[i] != 0:
                epsilon_core = Lum_m[i] / m[i]
            else:
                epsilon_core = 0
            Press_core = Press_m[i] + 2/3*np.pi*grav*rho_core**2*Rad_m[i]**2
            Temp_core = Press_core*mu*h_mass/(rho_core*boltz)

        elif m[i] <= 0:
            error_flag = 4
            rho_core = 0
            epsilon_core = 0
            Press_core = 0
            Temp_core = 0

        elif (Rad_m[i] < (0.02*R_surface)) and ((m[i] < (0.01*M_surface)) and ((Lum_m[i] < 0.1*L_surface))):
            rho_core = m[i] / (4/3*np.pi*Rad_m[i]**3)
            rho_max = 10 * (rho[i] / rho[i-1]) * rho[i]
            epsilon_core = Lum_m[i] / m[i]
            Press_core = Press_m[i] + 2/3*np.pi*grav*rho_core**2*Rad_m[i]**2
            Temp_core = Press_core*mu*h_mass / (rho_core*boltz)
            if (rho_core < rho[i]) or (rho_core > rho_max):
                error_flag = 1
            elif epsilon_core < epsilon[i]:
                error_flag = 2
            elif Temp_core < Temp_m[i]:
                error_flag = 3
            else:
                error_flag = 0

        # uncomment to break loop upon error detection
        # if error_flag != -1:
        #     istop = i
        #     break

        # determine whether to change step size
        if step_flag == 0 and (m[i] < (0.99*M_surface)):
            delta_m = -M_surface / 100
            step_flag = 1

        if step_flag == 1 and (delta_m >= (0.5*m[i])):
            delta_m = -M_surface / 5000
            step_flag = 2

        istop = i

    rho_core = m[istop] / (4/3*np.pi*Rad_m[istop]**3)
    epsilon_core = Lum_m[istop] / m[istop]
    Press_core = Press_m[istop] + 2/3*np.pi*grav*rho_core**2*Rad_m[istop]**2
    Temp_core = Press_core*mu*h_mass/(rho_core*boltz)

    if error_flag != 0:
        if error_flag == -1:
            print('The number of allowed shells has been exceeded.')

        if error_flag == 1:
            print('The core density seems unphysical.')
            print('The value calculated for the last zone was rho = ', rho[istop], ' gm/cm**3')
            print(rho_core, rho_max)

        if rho_core > 1e10:
            print('Yikes. You would need a degenerate')
            print('neutron gas and general relativity')
            print('to solve this core.')

        if error_flag == 2:
            print('The core epsilon seems unphysical.')
            print('The value calculated for the last zone was eps =', epsilon[istop], ' ergs/g/s')

        if error_flag == 3:
            print(' Your central temperature is too low.')
            print(' The value calculated for the last zone was T = ', Temp_m[istop], ' K')

        if error_flag == 4:
            print('Your star has a hole in the center.')

        if error_flag == 5:
            print('Your star has a negative central luminosity.')

        if error_flag == 6:
            print('You hit the center before the mass and/or')
            print('luminosity were depleted.')
    else:
        print('Remember to check your model for unphysical errors.')

    # print the central conditions and avoid format field overflows
    Rcrat = Rad_m[istop] / R_surface
    if Rcrat < -9.999:
        Rcrat = -9.999

    Mcrat = m[istop] / M_surface
    if Mcrat < -9.999:
        Mcrat = -9.999

    Lcrat = Lum_m[istop] / L_surface
    if Lcrat < -9.999:
        Lcrat = -9.999

    f = open('samwise_star.dat', 'w')

    f.write(' SAMWISE STAR MODEL\n')
    f.write(' ------------------\n')
    f.write(' Surface conditions:            Central conditions:\n')
    f.write(' Mtot = {0:13.6E} Msun          Mc/Mtot     = {1:12.5E}\n'.format(M_star, Mcrat))
    f.write(' Rtot = {0:13.6E} Rsun          Rc/Rtot     = {1:12.5E}\n'.format(R_star, Rcrat))
    f.write(' Ltot = {0:13.6E} Lsun          Lc/Ltot     = {1:12.5E}\n'.format(L_star, Lcrat))
    f.write(' Teff = {0:13.6E} K             Density     = {1:12.5E}\n'.format(T_star, rho_core))
    f.write(' X    = {0:13.6E}               Temperature = {1:12.5E}\n'.format(hydro, Temp_core))
    f.write(' Y    = {0:13.6E}               Pressure    = {1:12.5E} dynes/cm**2\n'.format(helium, Press_core))
    f.write(' Z    = {0:13.6E}               epsilon     = {1:12.5E} ergs/s/g\n'.format(metal, epsilon_core))

    f.write('Notes:\n')
    f.write(' (1) Mass is listed as Qm = 1.0 - M_star/M_tot.\n')
    f.write(' (2) Convective zones are denoted by c and radiative zones by r.\n')
    f.write(' (3) dlnP/dlnT may be limited to +99.9 or -99.9.\n')

    # print data for each shell
    f.write('   r        Qm         L       T        P        rho      kap      eps     dlPdlT\n')

    for ic in range(0, istop + 1):
        i = istop - ic
        Qm = 1 - m[i] / M_surface

        # label convective or radiative zones by c or r
        if dlPdlT[i] < gamma_index:
            etm = 'c'
        else:
            etm = 'r'

        # print warning flag (*) if abs(dlnP/dlnT) exceeds output limit
        if np.abs(dlPdlT[i]) > dlPlim:
            dlPdlT[i] = np.copysign(dlPlim, dlPdlT[i])
            clim = '*'
        else:
            clim = ' '

        s = '{0:7.2E} {1:7.2E} {2:7.2E} {3:7.2E} {4:7.2E} {5:7.2E} {6:7.2E} {7:6.2E}{8:1s}{9:1s} {10:5.1f}\n'.format(Rad_m[i], Qm, Lum_m[i], Temp_m[i], Press_m[i], rho[i], kappa[i], epsilon[i], clim, etm, dlPdlT[i])
        f.write(s)

    print('Done! The model has been stored in samwise_star.dat!')
    
    return error_flag, istop


def main():
    """
    Retrieve desired stellar parameters from user input.
    """
    M_star = float(input('Enter the mass of the star (solar mass):'))
    L_star = float(input('Enter the luminosity of the star (solar luminosity):'))
    T_star = float(input('Enter the temperature of the star (K):'))
    HYDROGEN = float(input('Enter the mass fraction of hydrogen:'))
    METALS = float(input('Enter the mass fraction of metals:'))
    HELIUM = 1 - HYDROGEN - METALS
    if HELIUM < 0:
        print('You must have X + Z <= 1. Please enter valid composition.')

    error_flag, istop = samwise_star(M_star, L_star, T_star, HYDROGEN, METALS)


main()
