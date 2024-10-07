import numpy as np
import matplotlib.pyplot as plt
import math

# Numerical values 

sigma = 1
epsilon = 5.9
alpha= 6.12* sigma**12
c= math.sqrt(epsilon*alpha/25)
E = 0.1
n = 1000
l = 1
h = 0.15

r = np.zeros(n)
W = np.zeros(n)
psi = np.zeros(n)

r[0] = h
r[1] = 2 * h
psi[0] = math.exp(-c * r[0]**(-5))
psi[1] = math.exp(-c * r[1]**(-5))

# defining functions 

# def V_lj(r):
#     return epsilon * ((sigma / r)**12 - 2 * (sigma / r)**6)


def F(r):
    return -6.12 * (E - l * (l + 1) /(6.12* r**2))


W[0] = (1 - (h**2) * (F(r[0]) / 12)) * psi[0]
W[1] = (1 - (h**2) * (F(r[1]) / 12)) * psi[1]

# Applying loop

for i in range(1, n - 1):
    r[i + 1] = (i + 2) * h
    W[i + 1] = 2 * W[i] - W[i - 1] + h**2 * F(r[i]) * psi[i]
    psi[i + 1] = W[i + 1] / (1 - (h**2) * F(r[i + 1]) / 12)



# Plot the results
plt.plot(r, psi)
plt.xlabel('r')
plt.ylabel('psi(r)')
plt.title('Wavefunction vs Radius')
plt.show()