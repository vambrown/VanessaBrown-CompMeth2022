"""Find distance to L1 point from Exercise 6.16 for Homework 4."""
import scipy.constants as sc


# Defining constants (astropy refused to cooperate)
M_E = 5.972e24  # Earth mass
M_M = 7.348e22  # Moon mass
A_V = 1 / (2.662e-6)**2  # inverse of angular velocity
EM_R = 3.844e8  # Earth-Moon distance


# Defining function and derivative for Newton's method
def orbit(dist):
    """Function showing distance from Earth
    to the L1 point."""
    return ((sc.G*M_E*A_V) / dist**2) - ((sc.G*M_M*A_V) / (EM_R - dist)**2)


def orbit_der(dist):
    """Derivative of above orbit function
    using the central difference method."""
    WIDTH = 1e-5
    return (orbit(dist + 0.5*WIDTH) - orbit(dist - 0.5*WIDTH)) / WIDTH


def newton(func, deriv, guess, error):
    """Using Newton's method to
    solve for the distance to L1."""
    while abs(func(guess)) > error:
        delta = func(guess) / deriv(guess)
        guess -= delta
    return guess


solution = newton(orbit, orbit_der, 1e8, 1e-4)
print("The distance from Earth to the L1 point is %d meters." % solution)
