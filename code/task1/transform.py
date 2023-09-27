import numpy as np
import scipy.fft
import matplotlib.pyplot as plt

def transform(f, N, start=0.0):
    # Create an array of N equidistant points
    x = np.linspace(start, start + 1.0, N)

    # Evaluate the function at the points
    f_values = f(x)

    print(f_values)
    print()
    # Compute the DFT using SciPy's FFT function
    f_hat = scipy.fft.fft(f_values)

    return x, f_values, f_hat

def plot_f_value_f_hat(f,N):

    x, f_values, f_hat = transform(f, N, -0.5)

    # Create a figure with 1 row and 2 columns of subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    # Plot the original function on the first subplot (axes[0])
    axes[0].plot(x, f_values, color="green", label="function")
    axes[0].set_title(f"Function {f.__name__} with N={N}")
    axes[0].legend()


    # Plot the real and imaginary parts of the DFT on the second subplot (axes[1])
    axes[1].plot(x, np.real(f_hat), color="red", label="real")
    axes[1].plot(x, np.imag(f_hat), color="blue", label="imaginary")
    axes[1].set_title(f"DFT of {f.__name__} with N={N}")
    axes[1].legend()
    print(x)
    print(f_values)

    # plt.tight_layout()
    plt.show()


def f1(x):
    return np.sin(8*np.pi*x)

def f2(x):
    return np.sin(32*np.pi*x) + np.cos(128*np.pi*x)

def f3(x):
    return x

def f4(x):
    return 1 - np.abs(x)


# plot_f_value_f_hat(f1, 5)
# plot_f_value_f_hat(f1,17)
# plot_f_value_f_hat(f1,257)

# plot_f_value_f_hat(f2, 5)
# plot_f_value_f_hat(f2,17)
plot_f_value_f_hat(f2,257 )


# plot_f_value_f_hat(f3, 5)
# plot_f_value_f_hat(f3,17)
# plot_f_value_f_hat(f3,257)

# plot_f_value_f_hat(f4, 5)
# plot_f_value_f_hat(f4,17)
# plot_f_value_f_hat(f4,257)
