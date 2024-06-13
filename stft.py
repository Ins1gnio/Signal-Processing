"""
Simple code for short time fourier transform (stft) of non-stationary sine/cosine signal in python. Created for signal processing related class, BME.
(c) AS. - 2024
"""

import matplotlib.pyplot as plt
import numpy as np


class signal_processing:
    def __init__(self):
        # parameter: generating non-stationary signal
        self.count_signal = 10  # define how many different signal
        self.amp_signal = [1, 2, 3, 4, 5, 5, 4, 3, 2, 1]  # define signal amplitude, the length should be the same with f_signal
        self.f_signal = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # define signal frequency (Hz), the length should be the same with amp_signal
        self.omega_signal = [2 * np.pi * f for f in self.f_signal]  # fundamental frequency (rad/s)
        self.overall_dur = 10  # overall signal duration (s)
        self.dur_signal = self.overall_dur / self.count_signal  # signal divided into 4 separate time (same duration)
        self.rate_signal = 100  # samples per seconds used in equation
        self.t = np.linspace(0, self.overall_dur, self.overall_dur * self.rate_signal)  # generate t (x axis)
        self.signal = np.zeros(self.overall_dur * self.rate_signal)
        self.label = ['Nan', 'NaN']

        # parameter: window
        self.win_width = 100    # define window width
        self.div, self.mod = divmod(self.overall_dur * self.rate_signal, self.win_width)    # allocate the x-axis
        self.win_val, self.signal_window = [np.zeros(self.overall_dur * self.rate_signal) for _ in range(2)]
        if self.mod == 0:
            self.win_segmenting = [[0 for _ in range(2)] for _ in range(self.div)]  # generate list if mod result = 0
        else:
            self.win_segmenting = [[0 for _ in range(2)] for _ in range(self.div + 1)]  # generate list if mod result != 0
        for i in range(len(self.win_segmenting)):   # fill the list
            self.win_segmenting[i][0] = self.win_width * i
            self.win_segmenting[i][1] = (self.win_width * (i + 1)) - 1
        if self.mod != 0:
            self.win_segmenting[-1][1] = self.win_segmenting[-1][0] + self.mod - 1
        # print(self.win_segmenting)

        # parameter: stft
        self.signal_stft = np.zeros((self.win_width, len(self.win_segmenting)), dtype=complex)
        self.freq = np.arange(self.win_width) * (self.rate_signal / self.win_width)
        self.time = np.linspace(0, self.overall_dur, len(self.win_segmenting))

        self.signal_generate()
        self.windowing()
        self.stft()
        self.plot()

    def signal_generate(self):
        for i in range(self.count_signal):
            for j in range(int(i * self.dur_signal * self.rate_signal), int((i + 1) * self.dur_signal * self.rate_signal)):
                # self.signal[j], self.label[0] = self.amp_signal[i] * np.sin(self.omega_signal[i] * self.t[j]), "Sine"  # generate sin wave
                self.signal[j], self.label[0] = self.amp_signal[i] * np.cos(self.omega_signal[i] * self.t[j]), "Cosine"  # generate cos wave

    def windowing(self):
        x_axis = np.arange(self.win_width)
        for i in range(len(self.win_segmenting)):
            for j in range(self.win_segmenting[i][0], self.win_segmenting[i][1] + 1):

                # choose the window
                # self.win_val[j], self.label[1] = 1, "Rectangular"   # rect. window
                self.win_val[j], self.label[1] = 1 - np.abs((x_axis[j - (i * self.win_width)] - (self.win_width / 2)) / (self.win_width / 2)), "Triangular"  # tri. window
                # self.win_val[j], self.label[1] = 0.54 - ((1 - 0.54) * np.cos((2 * np.pi * x_axis[j - (i * self.win_width)]) / self.win_width)), "Hamming"  # ham. window
                # self.win_val[j], self.label[1] = ((1 - 0.16) / 2) - 0.5 * np.cos((2 * np.pi * x_axis[j - (i * self.win_width)]) / self.win_width) + (0.16 / 2) * np.cos((4 * np.pi * x_axis[j - (i * self.win_width)]) / self.win_width), "Blackman"  # black. window
                # self.win_val[j], self.label[1] = ((0.21557895
                #                                    - 0.41663158 * np.cos((2 * np.pi * x_axis[j - (i * self.win_width)]) / self.win_width)
                #                                    + 0.277263158 * np.cos((4 * np.pi * x_axis[j - (i * self.win_width)]) / self.win_width))
                #                                   - 0.083578947 * np.cos((6 * np.pi * x_axis[j - (i * self.win_width)]) / self.win_width)
                #                                   + 0.006947368 * np.cos((8 * np.pi * x_axis[j - (i * self.win_width)]) / self.win_width)), "Flat-top"  # flat-top window

                self.signal_window[j] = self.signal[j] * self.win_val[j]    # multiply window and signal value

    def stft(self):
        for i in range(len(self.win_segmenting)):
            windowed_signal = self.signal_window[self.win_segmenting[i][0]:self.win_segmenting[i][1] + 1]

            # Short version of DFT
            n = np.arange(len(windowed_signal))
            k = n.reshape((len(windowed_signal), 1))
            comp_num = np.exp(-2j * np.pi * k * n / len(windowed_signal))  # compute real + imaginary parts
            dft = np.dot(comp_num, windowed_signal)  # signal source * complex num.
            self.signal_stft[:, i] = dft

    def plot(self):
        plt.figure("Short Time Fourier Transform (STFT)", figsize=(11, 6))
        plt.subplot(3, 1, 1)    # plot the signal
        plt.plot(self.t, self.signal, linewidth='2', label=self.label[0] + ' Wave')
        plt.xlabel("time (s)")
        plt.ylabel("amplitude")
        plt.xlim(0, self.overall_dur)
        plt.ylim(np.min(self.signal), np.max(self.signal))
        plt.legend(loc='upper right')

        plt.subplot(3, 1, 2)    # plot the window
        plt.plot(self.t, self.win_val, linewidth='2', label=self.label[1] + ' Window')
        plt.xlabel("time (s)")
        plt.ylabel("amplitude")
        plt.xlim(0, self.overall_dur)
        plt.ylim(np.min(self.win_val), np.max(self.win_val))
        plt.legend(loc='upper right')

        plt.subplot(3, 1, 3)    # plot the signal * window
        plt.plot(self.t, self.signal_window, linewidth='2', label='signal * window')
        plt.xlabel("time (s)")
        plt.ylabel("amplitude")
        plt.xlim(0, self.overall_dur)
        plt.ylim(np.min(self.signal_window), np.max(self.signal_window))
        plt.legend(loc='upper right')
        plt.show()

        plt.pcolormesh(self.time, self.freq, np.abs(self.signal_stft))  # plot stft value
        plt.ylim(0, self.rate_signal / 2)
        plt.xlabel('time (s)')
        plt.ylabel('frequency (Hz)')
        plt.colorbar(label='magnitude')
        plt.show()


if __name__ == '__main__':
    signal_processing()
