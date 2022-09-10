import numpy as np
import matplotlib.pyplot as plt

# the deltoid curve
theta1 = np.linspace(0, 2*np.pi, 2000)
x1 = 2*np.cos(theta1) + np.cos(2*theta1)
y1 = 2*np.sin(theta1) - np.sin(2*theta1)

# b) the Galilean spiral
theta2 = np.linspace(0, 10*np.pi, 2000)
r2 = theta2**2
x2 = r2*np.cos(theta2)
y2 = r2*np.sin(theta2)

# c) Fey's function
theta3 = np.linspace(0, 24*np.pi, 2000)
r3 = np.exp(np.cos(theta3)) - 2*np.cos(4*theta3) + (np.sin(theta3/12))**5
x3 = r3*np.cos(theta3)
y3 = r3*np.sin(theta3)

# plots
fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (7,3), dpi = 300)
plt.subplots_adjust(wspace = 0.5)
axes[0].plot(x1, y1, color = '#ffd343')
axes[0].set_box_aspect(1)
axes[0].set_title('Deltoid Curve')
axes[1].plot(x2, y2, color = '#9b59d0')
axes[1].set_box_aspect(1)
axes[1].set_title('Galilean Spiral')
axes[2].plot(x3, y3, color = 'k', linewidth = 0.7)
axes[2].set_box_aspect(1)
axes[2].set_title('Fey\'s Function')
fig.suptitle('Exercise 3.2: Curve Plotting', fontsize=16)
fig.text(0.5, 0.1, 'x', ha='center')
fig.text(0.05, 0.5, 'y', va='center', rotation='vertical')
plt.show()