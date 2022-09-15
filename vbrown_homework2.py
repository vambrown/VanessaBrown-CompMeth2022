import numpy as np
import matplotlib.pyplot as plt

# Using Simpson's rule to perform the integral of Exercise 5.3
x = np.arange(0, 3.1, 0.1)
def simpsons(N):
    def f(t):
        return np.exp(-t**2)
    a = 0
    b = x
    h = (b - a) / N
    sum_ends = f(a) + f(b)
    sum_odd = 0
    sum_even = 0
    for k in range(1, N, 2):
        sum_odd += f(a + k*h)
    for k in range(2, N, 2):
        sum_even += f(a + k*h)
    sum_odd *= 4
    sum_even *= 2
    sum = sum_ends + sum_odd + sum_even
    integral = (1/3)*sum*h
    return integral

# Making a graph of the intregral result using Simpson's rule and N = 1000
plt.plot(x, simpsons(1000), color = 'g', linestyle = 'dashed')
plt.xlabel('x')
plt.ylabel('E(x)')
plt.title('Exercise 5.3')
plt.show()