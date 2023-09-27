import numpy as np
import scipy.fft
import matplotlib.pyplot as plt

def transform(f, N, start=0.0, end=1.0):
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
    - x: ndarray
        An array containing the equidistant points at which the function
        was evaluated.
    """

    # Create an array of N equidistant points
    x = np.linspace(start, end, N, endpoint=True)

    # Evaluate the function at the points
    f_values = f(x)

    return x, f_values

def plot_f_and_f_hat(f, N):
    x_values, f_values = transform(f, N)

    # Compute the DFT using SciPy's FFT function
    f_hat = scipy.fft.fft(f_values)

    # Create a figure with 1 row and 2 columns of subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    # Plot the original function on the first subplot (axes[0])
    axes[0].plot(x_values, f_values, color="green", label="function")
    axes[0].set_title(f"Function {f.__name__} with N={N}")
    axes[0].legend()

    # Plot the real and imaginary parts of the DFT on the second subplot (axes[1])
    axes[1].plot(x_values, np.real(f_hat), color="red", label="real")
    axes[1].plot(x_values, np.imag(f_hat), color="blue", label="imaginary")
    axes[1].set_title(f"DFT of {f.__name__} with N={N}")
    axes[1].legend()

    # plt.tight_layout()
    plt.show()



def f1(x):
    return np.sin(8*np.pi*x)

def f2(x):
    return np.sin(32*np.pi) + np.cos(128*np.pi*x)

def f3(x):
    return x

def f4(x):
    return 1 - np.abs(x)

