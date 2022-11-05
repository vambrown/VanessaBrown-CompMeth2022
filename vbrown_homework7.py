"""Use importance sampling on the integral of Exercise 10.8 for HW7."""
import numpy as np


def g_of_x(x):
    """
    Integrand to be evaluated from
    0 to 1 for Exercise 10.8,

    x**(-0.5) / (np.exp(x) + 1),

    with divergent behavior removed.

    The weighting chosen to remove
    integrand's divergent behavior:
    x**(-0.5).
    """
    return 1 / (np.exp(x) + 1)


def p_of_x(z):
    """
    Probability distribution to use for
    importance sampling in divergent
    integrand.

    Transforms z values into x values.
    """
    return z**2


# a) use transformation method to get sample points
N = 1e6
z_values = np.random.rand(int(N))
x_values = p_of_x(z_values)

# b) evaluate the integral
SUM = 0
for xs in x_values:
    SUM += g_of_x(xs)
integral_value = (2 / N) * SUM

print("The value of the integral of Exercise 10.8 is %s." % integral_value)
