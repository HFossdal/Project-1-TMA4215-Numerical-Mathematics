import numpy as np
import scipy.fft
import matplotlib.pyplot as plt

def transform(f, N, start=0.0):
    """
    Transform a function f by evaluating it at N equidistant points within
    the interval [start, start+1).

    Parameters:
    - f: callable
        A function that takes a numeric input and returns a numeric output.
    - N: int
        The number of equidistant points at which to evaluate the function.
    - start: float, optional
        The starting point for the interval. Default is 0.0.

    Returns:
    - f_values: ndarray
        An array containing the values of the function f evaluated at N
        equidistant points within the specified interval.
    """

    # Create an array of N equidistant points
    x = np.linspace(start, start + 1.0, N, endpoint=False)

    # Evaluate the function at the points
    f_values = f(x)

    return f_values


def plot_f_value_f_hat(f, N):
    """
    Plot the real and imaginary parts of the DFT of a function f.

    Parameters:
    - f: callable
        A function that takes a numeric input and returns a numeric output.
    - N: int
        The number of equidistant points at which to evaluate the function.
    """

    # Compute the DFT using SciPy's FFT function
    f_values = transform(f, N)
    f_hat = scipy.fft.fft(f_values)

    plt.figure(figsize=(10, 4))

    # Plot the real and imaginary parts with different colors
    plt.plot(np.real(f_hat), color="red", label="real")
    plt.plot(np.imag(f_hat), color="blue", label="imaginary")

    # plot the function
    plt.plot(f_values, color="green", label="function")

    plt.title(f"Real and imaginary parts of the DFT of {f.__name__} with N={N}")
    plt.grid()
    plt.legend()
    plt.show()


def f1(x):
    return np.sin(8*np.pi*x)

def f2(x):
    return np.sin(32*np.pi) + np.cos(128*np.pi*x)

def f3(x):
    return x

def f4(x):
    return 1 - np.abs(x)


# plot_f_value_f_hat(f1, 5)
# plot_f_value_f_hat(f1,17)
# plot_f_value_f_hat(f1,257)

# plot_f_value_f_hat(f2, 5)
# plot_f_value_f_hat(f2,17)
# plot_f_value_f_hat(f2,257)

# plot_f_value_f_hat(f3, 5)
# plot_f_value_f_hat(f3,17)
# plot_f_value_f_hat(f3,257)

plot_f_value_f_hat(f4, 5)
plot_f_value_f_hat(f4,17)
plot_f_value_f_hat(f4,257)

