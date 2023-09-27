import transform
import scipy.fft
import numpy as np
import matplotlib.pyplot as plt

def plot_with_shift(f,N):

    x, f_values, f_hat = transform(f, N, -0.5)

    # Shift the DFT so that the zero frequency is in the middle
    f_hat_shift = scipy.fft.fftshift(f_hat)

    # Create a figure with 1 row and 2 columns of subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    # Plot the original function on the first subplot (axes[0])
    axes[0].plot(x, f_values, color="green", label="function")
    axes[0].set_title(f"Function {f.__name__} with N={N}")
    axes[0].legend()


    # Plot the real and imaginary parts of the DFT on the second subplot (axes[1])
    axes[1].plot(x, np.real(f_hat_shift), color="red", label="real")
    axes[1].plot(x, np.imag(f_hat_shift), color="blue", label="imaginary")
    axes[1].set_title(f"DFT of shiftet {f.__name__} with N={N}")
    axes[1].legend()

    # plt.tight_layout()
    plt.show()
