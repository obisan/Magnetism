__author__ = 'dubinets'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import time

TIME = 30
STEPS = 10**6
FPS = 60
BINING = int(STEPS / TIME / FPS)
DPI = 300
SIZE = 128

J = 1
magnetonBohr = 9.274e-24

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
    return - J * system[y, x] * \
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
    return np.sum(system) / (n * n)

#----------------------------------------------------------------------#
#   Calculate heat capacity
#----------------------------------------------------------------------#
def heatCapacity(n, T, system):
    sum = np.sum([energy(system, x, y) for x in list(range(SIZE)) for y in list(range(SIZE))])
    return (1.0 / (T**2)) * ( (sum**2)/n**2 - (sum / n**2)**2 )

#----------------------------------------------------------------------#
#   Calculate magnetic susceptibility
#----------------------------------------------------------------------#
def magneticSusceptibility(n, T, system):
    return (1.0 / (n * T)) * ((np.sum(system)**2)/n**2 - (np.sum(system)/n**2)**2)

#----------------------------------------------------------------------#
#   The Main monte carlo loop
#----------------------------------------------------------------------#
def simulation(T, system):
    systems = [ [ [0, magnetization(SIZE, system), heatCapacity(SIZE, T, system), magneticSusceptibility(SIZE, T, system)], system ] ]
    for step, x in enumerate(range(STEPS)):
        x = np.random.randint(0, SIZE)
        y = np.random.randint(0, SIZE)

        E = -2.0 * energy(system, x, y)

        if E <= 0.0:
            system[y, x] *= -1
        elif np.exp(-1.0 / T * E) > np.random.rand():
            system[y, x] *= -1

        if step%BINING==0:
            m   = magnetization(SIZE, system)                   # magnetic moment
            hc  = heatCapacity(SIZE, T, system)                 # heat capacity
            ms  = magneticSusceptibility(SIZE, T, system)       # magnetic susceptibility
            systems.append( [ [step, m, hc, ms], system.copy() ] )
    return systems

#----------------------------------------------------------------------#
#   Animate
#----------------------------------------------------------------------#
def draw(systems):
    X   = np.array( [x[0][0] for x in systems] )
    M   = np.array( [x[0][1] for x in systems] )
    HC  = np.array( [x[0][2] for x in systems] )
    MS  = np.array( [x[0][3] for x in systems] )

    fig = plt.figure()
    ax0 = fig.add_subplot(4, 1, 1)
    im0 = ax0.imshow(np.zeros((SIZE, SIZE)), vmin=-1, vmax=1)
    ax1 = fig.add_subplot(4, 1, 2, xlim=(X.min(), X.max()), ylim=(M.min(), M.max()))
    im1, = ax1.plot([], [])
    ax2 = fig.add_subplot(4, 1, 3, xlim=(X.min(), X.max()), ylim=(HC.min(), HC.max()))
    im2, = ax2.plot([], [])
    ax3 = fig.add_subplot(4, 1, 4, xlim=(X.min(), X.max()), ylim=(MS.min(), MS.max()))
    im3, = ax3.plot([], [])

    def animate(i):
        im0.set_data(systems[i][1])
        im1.set_data(X[:i], M[:i])  # update the data
        im2.set_data(X[:i], HC[:i])
        im3.set_data(X[:i], MS[:i])
        return im1,
    def init():
        im0.set_data([[]])
        im1.set_data([],[])
        return im1,

    print("Convert video...")
    t1 = time.clock()
    anim = manimation.FuncAnimation(fig, animate, frames=len(X), init_func=init, interval=10, blit=False)
    anim.save('Ising_model-Metropolis_algorithm-T=1-v.2.mp4', fps=FPS, extra_args=['-vcodec', 'libx264'])

    print("Convert video finished successful.")
    print("Total time of converting: %.3f sec." % (time.clock() - t1))

#----------------------------------------------------------------------#
#   Final image
#----------------------------------------------------------------------#
def image(systems):
    X = np.array( [x[0][0] for x in systems] )
    M = np.array( [x[0][1] for x in systems] )

    fig = plt.figure()
    fig.set
    ax0 = fig.add_subplot(2, 1, 1)
    im0 = ax0.imshow(systems[-1][1], vmin=-1, vmax=1)
    ax1 = fig.add_subplot(2, 1, 2, xlim=(X.min(), X.max()), ylim=(M.min(), M.max()))
    im1 = ax1.plot(X, M)

    plt.show()

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

    T = float(input("Choose the temperature for your simulation (0.1-10): "))

    print("Build system...")
    system = build_system()
    systems = simulation(T, system)
    print("Build system finished successful.")

    draw(systems)
    #image(systems)

if __name__ == "__main__":
    main()
