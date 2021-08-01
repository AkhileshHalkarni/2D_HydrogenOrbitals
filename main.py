import numpy as np
from scipy.special import eval_genlaguerre, lpmv, binom, factorial
import matplotlib.pyplot as plt

h_bar = 6.62607015e-34 / (2 * np.pi)  # reduced plank constant
me = 9.1093837015e-31  # electron's mass
eps0 = 8.85418782e-12  # Vacuum permittivity
e = 1.602176634e-19  # Electron charge

a = 0.529e-10


def compute_radial_hydrogen(n, l, r, a=0.529e-10):
    laguerre = eval_genlaguerre(n - l - 1, 2 * l + 1, (2 * r / (n * a)))
    sqrt_term = (2 / (n * a)) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l))
    exp_term = -r / (n * a)
    third_term = (2 * r / (n * a)) ** l

    return np.sqrt(sqrt_term) * np.exp(exp_term) * third_term * laguerre

def compute_spherical_harmonics_hydrogen(l, m, theta, phi):
    first_term = np.sqrt(((2 * l + 1) * factorial(l - m)) / (4 * np.pi * factorial(l + m))) * np.exp(1j * m * phi)
    legendre = lpmv(m, l, np.cos(theta))

    return first_term * legendre

def hydrogen_wave_function(x, n, l, m, a=0.529e-10):
    assert (len(x.shape) < 3), "X dimension must be at most 2"
    assert n > 0, "n must be greater than 0"
    assert l < n and l >= 0, "l must be between 0 and n-1"
    assert m <= l and m >= -l, "m must be between -l and l"

    theta = 0
    phi = 0
    if len(x.shape) == 1 or x.shape[1] == 1:
        r = x
    else:
        assert x.shape[1] < 4, "2nd dim of x must be at most 3"
        r = np.sum(x ** 2, axis=1)
        theta = np.arctan2(x[:, 1], x[:, 0])
        if x.shape[1] == 3:
            phi = np.arctan(np.sum(x[:, :2] ** 2, axis=1) / x[:, 2])

    hwf = compute_radial_hydrogen(n, l, r, a) * compute_spherical_harmonics_hydrogen(l, m, theta, phi)

    return hwf


def plot_hydrogen_orbitals(n, l, m, precision=800, posx=(-10, 10), posy=(-10, 10), posz=None):
    precisiony = int((max(posy) - min(posy)) / (max(posx) - min(posx)) * precision)

    # Creating grid points
    x = np.linspace(posx[0], posx[1], precision)
    y = np.linspace(posy[0], posy[1], precisiony)
    xx, yy = np.meshgrid(x, y)
    grid = np.vstack((xx.flatten(), yy.flatten())).T
    hwf = hydrogen_wave_function(grid, n, l, m, a=1).reshape(xx.shape)
    probs = np.abs(hwf) ** 0.7

    # Plotting
    plt.imshow(probs, cmap='magma')
    plt.axis('off')
    title = "Hydrogen orbitals with n=" + str(n) + ", l=" + str(l) + ", m=" + str(m)
    plt.title(title)
    plt.show()


n = int(input("Enter the value of n ::"))
l = int(input("Enter the value of l ::"))
m = int(input("Enter the value of m ::"))
plot_hydrogen_orbitals(n, l, m)


