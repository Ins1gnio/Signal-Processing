"""
Simple code for continuous wavelet transform (cwt) of non-stationary sine/cosine signal in python. Created for signal processing related class, BME.
Scale = f0 / freq. In the code below, to convert scale to freq, you need to adjust with sampling rate & tau, freq = scale / (fs * tau)
(c) AS. - 2024
"""

import matplotlib.pyplot as plt
import numpy as np


class signal_processing:
    def __init__(self):
        # parameter: generating non-stationary signal
        self.count_signal = 5  # define how many different signal
        self.amp_signal = [1, 1, 1, 1, 1]  # define signal amplitude, the length should be = self.count_signal
        self.f_signal = [1, 2, 3, 4, 5]  # define signal frequency (Hz), the length should be = self.count_signal
        self.omega_signal = [2 * np.pi * f for f in self.f_signal]  # fundamental frequency (rad/s)
        self.overall_dur = 5  # overall signal duration (s)
        self.dur_signal = self.overall_dur / self.count_signal  # signal divided into 4 separate time (same duration)
        self.rate_signal = 100  # samples per seconds used in equation
        self.t = np.linspace(0, self.overall_dur, self.overall_dur * self.rate_signal)  # generate t (x axis)
        self.signal = np.zeros(int(self.overall_dur * self.rate_signal))
        self.label = ['Nan', 'NaN']

        # parameter: cwt
        self.n_scale = 100
        self.scale = np.linspace(0.001, 0.1, self.n_scale)  # define the range of scales (start, stop, num)
        self.tau = 0.001  # define time increment
        self.f0 = 0.849  # define center frequency
        self.omega0 = 2 * np.pi * self.f0  # define center fund. frequency

        self.signal_generate()
        self.cwt()
        self.plot()

    def signal_generate(self):
        for i in range(self.count_signal):
            for j in range(int(i * self.dur_signal * self.rate_signal),
                           int((i + 1) * self.dur_signal * self.rate_signal)):
                # self.signal[j], self.label[0] = self.amp_signal[i] * np.sin(self.omega_signal[i] * self.t[j]), "Sine"  # generate sin wave
                self.signal[j], self.label[0] = self.amp_signal[i] * np.cos(
                    self.omega_signal[i] * self.t[j]), "Cosine"  # generate cos wave

    def cwt(self):
        cwt_ril, cwt_imag = [np.zeros((len(self.t), self.n_scale)) for _ in range(2)]
        t = np.arange(len(self.t)) * self.tau

        for i, scale in enumerate(self.scale):
            for j in range(len(self.t)):
                tt = (t - t[j]) / scale  # (time - shift) / scale
                wav = (1 / np.sqrt(scale)) * (np.pi ** -0.25) * np.exp(-(tt ** 2) / 2) * np.exp(
                    1j * self.omega0 * tt)  # cmorlet
                wav_sig = self.signal * wav  # conv. with signal
                cwt_ril[j, i] = np.sum(wav_sig.real)  # integrate for all times (real)
                cwt_imag[j, i] = np.sum(wav_sig.imag)  # integrate for all times (imag)

        self.cwt_asli = np.sqrt(cwt_ril ** 2 + cwt_imag ** 2)

    def rotate_matrix(self, m):     # rotate matrix 90 degree ccw
        return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]) - 1, -1, -1)]

    def plot(self):
        plt.figure("Non-stationary Signal", figsize=(11, 6))  # plot the signal
        plt.plot(self.t, self.signal, linewidth='2', label=self.label[0] + ' Wave')
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.xlim(0, self.overall_dur)
        plt.ylim(np.min(self.signal), np.max(self.signal))
        plt.legend(loc='upper right')
        plt.show()

        plt.figure("Continuous Wavelet Transform (CWT)", figsize=(10, 8))  # plot the cwt
        extent = [0, len(self.t), self.scale[0], self.scale[-1]]
        plt.imshow(np.abs(self.rotate_matrix(self.cwt_asli)), extent=extent, cmap='Greens', aspect='auto')
        plt.colorbar(label='Magnitude')
        plt.title('Continuous Wavelet Transform (CWT)')
        plt.xlabel('Time')
        plt.ylabel('Scale')
        plt.show()


if __name__ == '__main__':
    signal_processing()
