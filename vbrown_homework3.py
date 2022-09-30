"""Visualize electric potential/field from Exercise 5.21 for Homework 3."""
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# a) Electric potential

def e_potential(x_points, y_points):
    """
    Function for electric potential from two point charges.
    Takes grid coordinates, finds their distances to charges,
    and calculates potential at coordinates.
    """
    x_charge1 = 0.45
    x_charge2 = 0.55
    y_charges = 0.5

    r_point1 = np.sqrt((x_charge1 - x_points)**2 + (y_charges - y_points)**2)
    r_point2 = np.sqrt((x_charge2 - x_points)**2 + (y_charges - y_points)**2)

    v_point1 = sc.e / (4*np.pi*sc.epsilon_0*r_point1)
    v_point2 = (-sc.e) / (4*np.pi*sc.epsilon_0*r_point2)

    return v_point1 + v_point2


# Making a grid on which to get electric potential values
X, Y = np.meshgrid(np.arange(0, 1.01, 0.01), np.arange(0, 1.01, 0.01))
v_values = e_potential(X, Y)

# b) Electric field

# Using the central difference method to derive the field
WIDTH = 1e-5
E_x = (e_potential(X + 0.5*WIDTH, Y) - e_potential(X - 0.5*WIDTH, Y)) / WIDTH
E_y = (e_potential(X, Y + 0.5*WIDTH) - e_potential(X, Y - 0.5*WIDTH)) / WIDTH

# Creating contour and quiver plots
fig, axes = plt.subplots(nrows=1, ncols=2)
fig.suptitle('Exercise 5.21: Charge Distribution', fontsize=20)
cf = axes[0].contourf(X, Y, v_values, 20)
axes[0].set_title('Electric Potential', fontsize=16)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_aspect('equal')
color = 2 * np.log(np.hypot(E_x, E_y))
axes[1].streamplot(X, Y, E_x, E_y, color=color, linewidth=1, arrowstyle='->')
axes[1].set_title('Electric Field', fontsize=16)
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_aspect('equal')
axes[1].add_artist(Circle((0.45, 0.5), 0.005, color='yellow'))
axes[1].add_artist(Circle((0.55, 0.5), 0.005, color='indigo'))
plt.show()
