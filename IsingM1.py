__author__ = 'dubinets'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import time

TIME = 30
STEPS = 10**5
BINING = 16
FPS = STEPS / TIME / BINING
DPI = 300
SIZE = 128

J = -1
Bohrmagneton = 9.274e-24

#----------------------------------------------------------------------#
#   Check periodic boundary conditions
#----------------------------------------------------------------------#
def bc(i):
    if i+1 > SIZE-1:
        return 0
    if i-1 < 0:
        return SIZE-1
    else:
        return i

#----------------------------------------------------------------------#
#   Calculate internal energy
#----------------------------------------------------------------------#
def energy(system, x, y):
    return J * system[y, x] * \
           (system[bc(y - 1), x] +
            system[bc(y + 1), x] +
            system[y, bc(x - 1)] +
            system[y, bc(x + 1)])

#----------------------------------------------------------------------#
#   Build the system
#----------------------------------------------------------------------#
def build_system():
    system = np.random.random_integers(0,1,(SIZE,SIZE))
    system[system==0] = -1
    return system

#----------------------------------------------------------------------#
#   Calculate magnetization
#----------------------------------------------------------------------#
def magnetization(n, system):
    return np.sum(system) / n

#----------------------------------------------------------------------#
#   The Main monte carlo loop
#----------------------------------------------------------------------#
def simulation(T, system):
    systems = [ [ [0, magnetization(SIZE, system)], system ] ]
    for step, x in enumerate(range(STEPS)):
        x = np.random.randint(0, SIZE)
        y = np.random.randint(0, SIZE)

        E = -2.0 * energy(system, x, y)

        if E <= 0.0:
            system[y, x] *= -1
        elif np.exp(-1.0 / T * E) > np.random.rand():
            system[y, x] *= -1

        if step%BINING==0:
            systems.append( [ [step, magnetization(SIZE, system)], system.copy() ] )
    return systems

#----------------------------------------------------------------------#
#   Run the menu for the monte carlo simulation
#----------------------------------------------------------------------#
def main():
    print('='*70)
    print('\tMonte Carlo Statistics for an Ising model with')
    print('\t\tperiodic boundary conditions')
    print('='*70)

    print("frames: ", STEPS)
    print("time: ", TIME)
    print("fps: ", FPS)
    print("dpi: ", DPI)
    print('='*70)

    T = 2 #float(input("Choose the temperature for your simulation (0.1-10): "))

    print("Build system...")
    system = build_system()
    systems = simulation(T, system)

    print("Build system finished successful.")

    X = np.array( [x[0][0] for x in systems] )
    M = np.array( [x[0][1] for x in systems] )

    fig = plt.figure()
    ax0 = fig.add_subplot(2, 1, 1)
    im0 = ax0.imshow(np.zeros((SIZE, SIZE)), vmin=-1, vmax=1)
    ax1 = fig.add_subplot(2, 1, 2, xlim=(X.min(), X.max()), ylim=(M.min(), M.max()))
    im1, = ax1.plot([], [])

    def animate(i):
        im0.set_data(systems[i][1])
        im1.set_data(X[:i], M[:i])  # update the data
        return im1,
    def init():
        im0.set_data([[]])
        im1.set_data([],[])
        return im1,

    print("Convert video...")
    t1 = time.clock()
    anim = manimation.FuncAnimation(fig, animate, frames=len(X), init_func=init, interval=10, blit=False)
    anim.save('test_plot.mp4', fps=FPS, extra_args=['-vcodec', 'libx264'])

    print("Convert video finished successful.")
    print("Total time of converting: %.3f sec." % (time.clock() - t1))

if __name__ == "__main__":
    main()
