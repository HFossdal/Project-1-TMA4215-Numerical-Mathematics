import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as special
import scipy.fft
import scipy.signal as signal

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





csv_file = r"data/project1-1e-data.csv"


def file_to_signal_array(file_path):
    # Open and read the CSV file
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        # skip the header
        next(csv_reader)
        # create an empty array
        signal = []
        # add the values from the CSV file to the array
        for row in csv_reader:
            signal.append(float(row[1]))
    return np.array(signal)


signal = file_to_signal_array(csv_file)

# length of signal
length = len(file_to_signal_array(csv_file))

x = np.arange(length)


# Constructing the dirichlet kernel

# Number of samples
N_samples = length

# Define a function to compute the Dirichlet kernel
def dirichlet(x):
    return special.diric(x, N_samples)

# Use the transform function to discretize the dirichlet kernel
dirichlet_values =transform(dirichlet, N_samples, -0.5)[1]

# define the second kernel, h
h = np.zeros(length)
h[0] = -1
h[1] = 2
h[2] = -1

# convolution of signal and dirichlet kernel
convolution_dirichlet92 = np.convolve(dirichlet_values, signal)

# convolution of signal and h
convolution_h = np.convolve(h, signal)

# Plot the original signal, dirichlet kernel, and the convolution result
plt.figure(figsize=(12, 6))
plt.subplot(321)
plt.plot(signal, color="blue", label="signal")
plt.title("Original Signal")
plt.legend()

plt.subplot(323)
plt.plot(dirichlet_values, color="red", label="dirichlet kernel")
plt.title("Dirichlet Kernel")
plt.legend()

plt.subplot(325)
plt.plot(convolution_dirichlet92, color="green", label="convolution")
plt.title("Convolution of signal with Dirichlet Kernel")
plt.legend()



# Plot the original signal, h, and the convolution result
plt.subplot(322)
plt.plot(signal, color="blue", label="signal")
plt.title("Original Signal")
plt.legend()

plt.subplot(324)
plt.plot(h, color="red", label="h")
plt.title("h")
plt.legend()

plt.subplot(326)
plt.plot(convolution_h, color="green", label="convolution")
plt.title("Convolution of signal with h")
plt.legend()


plt.tight_layout()
plt.show()

