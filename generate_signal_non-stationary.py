"""
Simple code for sine/cosine non-stationary signal generation in python. Created for signal processing related class, BME.
(c) AS. - 2024
"""

import matplotlib.pyplot as plt
import numpy as np


class signal_processing:
    def __init__(self):
        self.signal_generate()
        self.plot()

    def signal_generate(self):
        count_signal = 6  # define how many different signal
        amp_signal = [1, 2, 3, 4, 5, 6]  # define signal amplitude
        f_signal = [1, 2, 3, 4, 5, 6]  # define signal frequency (Hz)
        omega_signal = [2 * np.pi * f for f in f_signal]  # fundamental frequency (rad/s)
        self.overall_dur = 10  # overall signal duration (s)
        self.dur_signal = self.overall_dur / count_signal  # signal divided into 4 separate time (same duration)
        rate_signal = 1000  # samples per seconds used in equation
        self.t = np.linspace(0, self.overall_dur, self.overall_dur * rate_signal)  # generate t (x axis)
        self.signal = np.zeros(self.overall_dur * rate_signal)
        for i in range(count_signal):
            for j in range(int(i * self.dur_signal * rate_signal), int((i + 1) * self.dur_signal * rate_signal)):
                self.signal[j] = amp_signal[i] * np.sin(omega_signal[i] * self.t[j])  # generate sin wave
                # self.signal[j] = amp_signal[i] * np.cos(omega_signal[i] * self.t[j])  # generate cos wave

    def plot(self):  # plot the signal
        plt.figure(figsize=(11, 5))
        plt.plot(self.t, self.signal, linewidth='2', label='Sine Wave')
        plt.xlabel("time (s)")
        plt.ylabel("amplitude")
        plt.xlim(0, self.overall_dur)
        plt.ylim(np.min(self.signal), np.max(self.signal))
        plt.legend(loc='upper right')
        plt.show()


if __name__ == '__main__':
    signal_processing()
