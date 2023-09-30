import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft
import scipy.signal as signal

def file_to_signal_array(file_path):
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        signal = []
        for row in csv_reader:
            signal.append(float(row[1]))
    return np.array(signal)

csv_file = r'data/project1-1e-data.csv'
signal = file_to_signal_array(csv_file)

# Number of samples
N_samples = len(signal)

# Define the Dirichlet kernel in the frequency domain
n = 92 # the n in the Dirichlet kernel


# Define the h kernel
h_kernel = np.zeros(N_samples)
h_kernel[0] = -1
h_kernel[1] = 2
h_kernel[2] = -1

# Compute the Fourier transforms of the signal, Dirichlet kernel, and h kernel
signal_fft = scipy.fft.fft(signal)
dirichlet_kernel_fft = np.zeros(N_samples)
dirichlet_kernel_fft[int(N_samples/2-n-1):int(N_samples/2+n)] = 1
h_kernel_fft = scipy.fft.fft(h_kernel, N_samples)
dirichlet_kernel_fft = scipy.fft.fftshift(dirichlet_kernel_fft)


# Perform convolution in the frequency domain
convolution_dirichlet_fft = signal_fft * dirichlet_kernel_fft
convolution_h_fft = signal_fft * h_kernel_fft


# Compute the inverse Fourier transforms to get the convolution results
convolution_dirichlet = np.real(scipy.fft.ifft(convolution_dirichlet_fft))
convolution_h = np.real(scipy.fft.ifft(convolution_h_fft))


# Plotting
x_vals = np.linspace(0, 1, N_samples)

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(x_vals, signal, color="blue", label="signal")
plt.title("Original signal")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x_vals, signal, color="blue", label="signal")
plt.title("Original signal")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x_vals, convolution_dirichlet, color="red", label="convolution")
plt.title("Convolution with Dirichlet kernel")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(x_vals, convolution_h, color="red", label="convolution")
plt.title("Convolution with h kernel")
plt.legend()

plt.tight_layout()
plt.show()
