"""
Simple code for sine/cosine signal generation in python. Created for signal processing related class, BME.
(c) AS. - 2024
"""

import matplotlib.pyplot as plt
import numpy as np


class signal_processing:
    def __init__(self):
        self.signal_generate()
        self.plot()

    def signal_generate(self):
        amp_signal = 2  # signal amplitude
        f_signal = 1  # signal frequency (Hz)
        omega_signal = 2 * np.pi * f_signal  # fundamental frequency (rad/s)
        self.dur_signal = 10  # signal duration (s)
        rate_signal = 1000  # samples in equation per second
        self.t = np.linspace(0, self.dur_signal, rate_signal * self.dur_signal)  # generate t (x axis)
        self.signal = amp_signal * np.sin(omega_signal * self.t)  # generate sin wave
        # self.signal = amp_signal * np.cos(omega_signal * self.t)  # generate cos wave

    def plot(self):     # plot the signal
        plt.figure(figsize=(11, 5))
        plt.plot(self.t, self.signal, linewidth='2', label='Sine Wave')
        plt.xlabel("time (s)")
        plt.ylabel("amplitude")
        plt.xlim(0, self.dur_signal)
        plt.ylim(np.min(self.signal), np.max(self.signal))
        plt.legend(loc='upper right')
        plt.show()


if __name__ == '__main__':
    signal_processing()