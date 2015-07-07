import numpy as np
import matplotlib.pyplot as plt

SIZE = 150
STEPS = 10**6

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
def energy(system, N, M):
    return -1 * system[N,M] * \
           (system[bc(N-1), M] +
            system[bc(N+1), M] +
            system[N, bc(M-1)] +
            system[N, bc(M+1)])

#----------------------------------------------------------------------#
#   Build the system
#----------------------------------------------------------------------#
def build_system():
    system = np.random.random_integers(0,1,(SIZE,SIZE))
    system[system==0] =- 1
    return system

#----------------------------------------------------------------------#
#   The Main monte carlo loop
#----------------------------------------------------------------------#
def simulation(T):
    system = build_system()

    for step, x in enumerate(range(STEPS)):
        M = np.random.randint(0,SIZE)
        N = np.random.randint(0,SIZE)

        E = -2. * energy(system, N, M)

        if E <= 0.:
            system[N,M] *= -1
        elif np.exp(-1. / T * E) > np.random.rand():
            system[N,M] *= -1

    #plt.imshow(system)
    #plt.show()

    return system

#----------------------------------------------------------------------#
#   Run the menu for the monte carlo simulation
#----------------------------------------------------------------------#
def main():
    print('='*70)
    print('\tMonte Carlo Statistics for an ising model with')
    print('\t\tperiodic boundary conditions')
    print('='*70)

    print("Choose the temperature for your simulation (0.1-10)")

    data = []

    x = 0.1
    while(x <= 10):
        T = float(x)
        data.append([T, (sum(sum(np.array(simulation(T))))) / SIZE ])
        x += 0.1

    temperature = np.array([row[0] for row in data])
    magnetization = np.array([row[1] for row in data])

    print(temperature)
    print(magnetization)

    plt.plot(temperature, magnetization)
    plt.show()

main()