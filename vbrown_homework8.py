"""Plot trajectory with air resistance of Exercise 8.7 for Homework 8."""
import argparse
import numpy as np
import matplotlib.pyplot as plt


def f(r, constants):
    """
    Equations of motions for position (x, y) of
    a spherical cannonball including air resistance.
    """
    x = r[0]  # x position
    y = r[1]  # y position
    x_v = r[2]  # x velocity
    y_v = r[3]  # y velocity

    # x and y acceleration
    x_a = (-1) * constants * x_v * np.sqrt(x_v**2 + y_v**2)
    y_a = (-1 * GRAV) - constants * y_v * np.sqrt(x_v**2 + y_v**2)

    return np.array([x_v, y_v, x_a, y_a])


# constants
RADIUS = 0.08  # m
RHO = 1.22  # kg / m^3
DRAG = 0.47  # coefficient for sphere
GRAV = 9.81  # m / s^2

# conditions
ANGLE = 30 * np.pi / 180  # degrees to radians
VEL_0 = 100  # m/s
X_VEL_0 = VEL_0 * np.cos(ANGLE)  # initial x velocity
Y_VEL_0 = VEL_0 * np.sin(ANGLE)  # initial y velocity
X_POS_0 = 0  # initial x position
Y_POS_0 = 0  # initial y position

START = 0.0
STOP = 10.0
STEP = 0.01


def trajectory(mass):
    """Fourth-order Runge-Kutta solution for cannonball trajectory
    with air resistance."""
    const = (np.pi * RADIUS**2 * RHO * DRAG) / (2.0 * mass)

    times = np.arange(START, STOP + STEP, STEP)
    vector = np.zeros((len(times), 4))
    vector[0, :] = [X_POS_0, Y_POS_0, X_VEL_0, Y_VEL_0]

    for i in range(1, len(times)):
        k1 = f(vector[i-1, :], const)
        k2 = f(vector[i-1, :] + (STEP/2)*k1, const)
        k3 = f(vector[i-1, :] + (STEP/2)*k2, const)
        k4 = f(vector[i-1, :] + STEP*k3, const)
        vector[i, :] = vector[i-1, :] + (STEP / 6)*(k1 + 2*k2 + 2*k3 + k4)

    return vector[:i]


def cannon_plot(mass):
    cannonball = trajectory(mass)
    cannonball_150 = trajectory((1.25*mass))
    cannonball_80 = trajectory((0.8*mass))
    cannonball_50 = trajectory((0.5*mass))
    cannonball_10 = trajectory((0.1*mass))

    plt.plot(cannonball[:, 0], cannonball[:, 1], 'k', label="Cannonball Mass")
    plt.plot(cannonball_150[:, 0], cannonball_150[:, 1], '--', label="150% Cannonball Mass")
    plt.plot(cannonball_80[:, 0], cannonball_80[:, 1], '--', label="80% Cannonball Mass")
    plt.plot(cannonball_50[:, 0], cannonball_50[:, 1], '--', label="50% Cannonball Mass")
    plt.plot(cannonball_10[:, 0], cannonball_10[:, 1], '--', label="10% Cannonball Mass")
    plt.legend()
    plt.title('Trajectory of Cannonball with Air Resistance')
    plt.xlabel('Distance (m)')
    plt.ylabel('Height (m)')
    plt.ylim(bottom=0)
    plt.show()

# setting up the argparse
ARG_DESC = '''\
        Fire the cannons!
        --------------------------------
            This program plots the trajectory of a
            cannonball (including air resistance) with
            trajectories of similar-mass cannonballs
            for comparison.
        '''
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=ARG_DESC)
parser.add_argument("mass", type=int,
                    help="Enter a mass (kg) for the cannonball.")
args = parser.parse_args()

cannon_mass = args.mass
cannon_plot(cannon_mass)
