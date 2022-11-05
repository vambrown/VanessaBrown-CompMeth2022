"""Visualizing Brownian motion of Exercise 10.3 for HW6."""
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np


def random_walk(steps, grid_length):
    """
    Creates an animation of Brownian motion.

    Parameters
    ----------
    steps : int
        Number of random steps for particle to perform.

    grid_size : int
        Side length of grid confining particle motion.

    Returns
    -------
    An animation of a particle random-walking for a specified step
    number across a grid with specified side length.
    """

    # set grid size and create i, j positions on grid
    i = grid_length // 2
    j = grid_length // 2

    positions = []  # positions of particle at each step

    # randomly decide how to move as long as
    # the particle is inside the grid
    step = 0
    while step < steps:
        if i == grid_length:
            if np.random.rand() <= 0.3:
                i -= 1
            elif np.random.rand() <= 0.6:
                j -= 1
            else:
                j += 1
            step += 1
        elif j == grid_length:
            if np.random.rand() <= 0.3:
                j -= 1
            elif np.random.rand() <= 0.6:
                i -= 1
            else:
                i += 1
            step += 1
        elif i == 0:
            if np.random.rand() <= 0.3:
                i += 1
            elif np.random.rand() <= 0.6:
                j -= 1
            else:
                j += 1
            step += 1
        elif j == 0:
            if np.random.rand() <= 0.3:
                j += 1
            elif np.random.rand() <= 0.6:
                i -= 1
            else:
                i += 1
            step += 1
        else:
            if np.random.rand() <= 0.25:
                i -= 1
                positions.append([i, j])
            elif np.random.rand() <= 0.5:
                i += 1
                positions.append([i, j])
            elif np.random.rand() <= 0.75:
                j -= 1
                positions.append([i, j])
            else:
                j += 1
                positions.append([i, j])
            step += 1

    # setting up the animation
    fig = plt.figure()
    ax = plt.axes(xlim=(0, grid_length), ylim=(0, grid_length))
    plt.title("Brownian Motion")
    plt.xlabel("Grid Length (x)")
    plt.ylabel("Grid Length (y)")
    line, = ax.plot([], [], lw=2)
    i_pos = []
    j_pos = []

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        i_pos.append(positions[i][0])
        j_pos.append(positions[i][1])
        line.set_data(i_pos, j_pos)
        return line,

    anim = ani.FuncAnimation(fig, animate, init_func=init,
                             frames=100, interval=20, blit=True)

    writergif = ani.PillowWriter(fps=20)
    anim.save('VB_Brownian_Motion.gif', writer=writergif)


# setting up the argparse
ARG_DESC = '''\
        Let's simulate Brownian motion!
        --------------------------------
            This program creates a movie of a particle's
            random walk through a specified number of steps
            on a grid of specified length.
        '''
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=ARG_DESC)
parser.add_argument("step", type=int,
                    help="Enter number of steps for particle to perform.")
parser.add_argument("grid", type=int,
                    help="Enter length of grid to confine particle motion.")
args = parser.parse_args()

step_size = args.step
grid_size = args.grid
random_walk(step_size, grid_size)
