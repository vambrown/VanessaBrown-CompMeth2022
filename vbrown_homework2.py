"""Perform and graph integral from Exercise 5.3 for Homework 2."""
import numpy as np
import matplotlib.pyplot as plt


# Performing the integral of Exercise 5.3
x = np.arange(0, 3.1, 0.1)


def simpsons(steps):
    """Evaluates an integrand using Simpson's rule."""
    def exp_function(power):
        return np.exp(-power**2)
    lower_lim = 0
    upper_lim = x
    step_width = (upper_lim - lower_lim) / steps
    summation_ends = exp_function(lower_lim) + exp_function(lower_lim)
    summation_odd = 0
    summation_even = 0
    for k in range(1, steps, 2):
        summation_odd += exp_function(lower_lim + k*step_width)
    for k in range(2, steps, 2):
        summation_even += exp_function(lower_lim + k*step_width)
    summation_odd *= 4
    summation_even *= 2
    summation_all = summation_ends + summation_odd + summation_even
    integral = (1/3)*summation_all*step_width
    return integral


# Making a graph of the intregral result using Simpson's rule and N = 1000
plt.plot(x, simpsons(1000), color='g', linestyle='dashed')
plt.xlabel('x')
plt.ylabel('E(x)')
plt.title('Exercise 5.3')
plt.show()
