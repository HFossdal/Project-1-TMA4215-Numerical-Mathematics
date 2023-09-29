import scipy.special as special
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

def transform(f, N, start=0.0):
    # Create an array of N equidistant points
    x = np.linspace(start, start + 1.0, N)

    # Evaluate the function at the points
    f_values = f(x)

    # Compute the DFT using SciPy's FFT function
    f_hat = np.fft.fft(f_values)

    return x, f_values, f_hat

def f2(x):
    return np.sin(32*np.pi*x) + np.cos(128*np.pi*x)

N = 512

n = 48

def dirichlet(x):
    return special.diric(2 * np.pi*x, n)

x_values, f2_values, f2_hat = transform(f2, N, -0.5)

dirichlet_values = transform(dirichlet, N, -0.5)[1]

convolution = scipy.signal.convolve(f2_values, dirichlet_values, mode='same')



plt.figure(figsize=(12, 6))
plt.subplot(311)
plt.plot(x_values, f2_values, color="blue", label="f2(x)")
plt.title("Original Function f2(x)")
plt.legend()

plt.subplot(312)
plt.plot(x_values, dirichlet_values, color="red", label="Dirichlet Kernel")
plt.title("Dirichlet Kernel")
plt.legend()

plt.subplot(313)
plt.plot(x_values, convolution, color="green", label="Convolution")
plt.title("Convolution of f2 with Dirichlet Kernel")
plt.legend()

plt.tight_layout()
plt.show()



