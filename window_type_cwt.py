"""
Simple code for wavelet function in python. Created for signal processing related class, BME.
(c) AS. - 2024
"""

import numpy as np
import matplotlib.pyplot as plt


class main_window:
    def __init__(self):
        self.dur = 100  # overall duration
        self.rate = 1000  # data rate per 1 seq.
        self.shift = 50  # shifting parameter
        self.scale = 10  # scaling parameter
        self.x_axis = np.linspace(0, self.dur, self.dur * self.rate)  # x-axis

        self.win_real, self.win_imag, self.label = self.win_cmorlet()  # choose your window here (complex wavelet)
        # self.win_real, self.label = self.win_poisson()  # choose your window here (real only)
        self.plot()  # plot

    def win_cmorlet(self):   # complex mother wavelet (morlet)
        f0 = 0.849  # determine the amount of wave in the envelope, default is 0.849
        omega0 = 2 * np.pi * f0  # fund. freq.
        cmorlet = np.pi ** -0.25 * np.exp(-0.5 * ((self.x_axis - self.shift) / self.scale) ** 2) * np.exp(
            -1j * omega0 * ((self.x_axis - self.shift) / self.scale))  # cmorlet equation
        return np.real(cmorlet), np.imag(cmorlet), "C.Morlet"

    def win_cshannon(self):   # complex shannon wavelet
        cshannon = (np.sinc(((self.x_axis - self.shift) / self.scale))
                    * np.exp(-2 * np.pi * 1j * ((self.x_axis - self.shift) / self.scale)))    # cshannon equation
        return np.real(cshannon), np.imag(cshannon), "C.Shannon"

    def win_cpoisson(self):   # complex poisson wavelet
        n = 4
        cpoisson = (1 / (2 * np.pi)) * ((1 - (1j * ((self.x_axis - self.shift) / self.scale))) ** (-1 * (n + 1)))   # cpoisson equation
        return np.real(cpoisson), np.imag(cpoisson), "C.Poisson"

    def win_ricker(self):   # ricker / mexican-hat wavelet
        std = 1
        ricker = ((2 / (np.sqrt(3 * std) * (np.pi ** 0.25))) * (1 - ((((self.x_axis - self.shift) / self.scale) / std) ** 2))
                  * np.exp(-1 * (((self.x_axis - self.shift) / self.scale) ** 2) / (2 * (std ** 2))))     # ricker equation
        return ricker, "Mexican-Hat"

    def win_shannon(self):   # shannon wavelet
        shannon = 2 * np.sinc(2 * ((self.x_axis - self.shift) / self.scale)) - np.sinc(((self.x_axis - self.shift) / self.scale))   # shannon equation
        return shannon, "Shannon"

    def win_poisson(self):   # poisson wavelet
        poisson = (1 / np.pi) * ((1 - (((self.x_axis - self.shift) / self.scale) ** 2)) / (1 + (((self.x_axis - self.shift) / self.scale) ** 2)) ** 2)      # poisson equation
        return poisson, "Poisson"

    def plot(self):
        plt.plot(self.x_axis, self.win_real, 'r', linewidth=2, label='Real Part')   # plot real part
        plt.plot(self.x_axis, self.win_imag, 'b', linewidth=2, label='Imaginary Part')    # plot imaginary part (applicable on complex wavelet)
        plt.xlim(0, self.dur)
        plt.ylim(-1, 1)
        plt.legend(loc='upper right')
        plt.xlabel('t')
        plt.ylabel(r'$\psi(t)$')
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    main_window()
