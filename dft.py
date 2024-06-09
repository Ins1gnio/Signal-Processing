"""
Simple code to do Discrete Fourier Transform (DFT) in python. Created for signal processing related class, BME.
(c) AS. - 2024
"""

import matplotlib.pyplot as plt
import numpy as np


class signal_processing:
    def __init__(self):
        self.signal_generate()
        self.dft()
        self.plot()

    def signal_generate(self):
        amp_signal = 2  # signal amplitude
        f_signal = 5    # signal frequency (Hz)
        omega_signal = 2 * np.pi * f_signal  # fundamental frequency (rad/s)
        self.dur_signal = 5  # signal duration (s)
        rate_signal = 1000  # sample in equation per second
        self.n_signal = rate_signal * self.dur_signal  # sample count
        self.t = np.linspace(0, self.dur_signal, self.n_signal)  # generate t (x axis)
        self.signal = amp_signal * np.sin(omega_signal * self.t)  # generate sin wave
        # self.signal = amp_signal * np.cos(omega_signal * self.t)  # generate cos wave

    def dft(self):
        # Long version of DFT
        # real, imag = np.zeros(self.n_signal), np.zeros(self.n_signal)
        # for k in range(self.n_signal):
        #     real[k] = 0
        #     imag[k] = 0
        #     for n in range(self.n_signal):
        #         real[k] += self.signal[n] * np.cos((2 * np.pi * k * n) / self.n_signal)   # compute real part
        #         imag[k] += self.signal[n] * np.sin((2 * np.pi * k * n) / self.n_signal)   # compute imaginary part
        # self.signal_dft = real - 1j * imag  # combine real + imaginary parts
        # self.freq = np.arange(self.n_signal) * (1 / self.dur_signal)     # determine frequency resolution for x-axis

        # Short version of DFT
        n = np.arange(self.n_signal)
        k = n.reshape((self.n_signal, 1))
        comp_num = np.exp(-2j * np.pi * k * n / self.n_signal)  # compute real + imaginary parts
        self.signal_dft = np.dot(comp_num, self.signal)     # signal source * complex num.
        self.freq = np.arange(self.n_signal) * (1 / self.dur_signal)    # determine frequency resolution for x-axis

    def plot(self):  # plot the signal
        plt.figure(figsize=(11, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.t, self.signal, linewidth='2', label='Sine Wave')
        plt.xlabel("time (s)")
        plt.ylabel("amplitude")
        plt.xlim(0, self.dur_signal)
        plt.ylim(np.min(self.signal), np.max(self.signal))
        plt.legend(loc='upper right')

        plt.subplot(1, 2, 2)    # plot the DFT
        plt.stem(self.freq, np.abs(self.signal_dft), 'b', markerfmt=" ", basefmt="-b", label='Magnitude')
        plt.xlabel("freq (Hz)")
        plt.ylabel("magnitude")
        plt.xlim(0, self.freq[-1] / 2)
        plt.ylim(-100, np.max(np.abs(self.signal_dft)) + 100)
        plt.legend(loc='upper right')
        plt.show()


if __name__ == '__main__':
    signal_processing()
