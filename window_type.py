"""
Simple code for window function in python. Created for signal processing related class, BME.
(c) AS. - 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import math


class main_window:
    def __init__(self):
        self.x_start = 0
        self.x_finish = 100
        self.n_data = self.x_finish - self.x_start  # data count
        self.x_axis = np.arange(self.n_data)    # x axis
        self.y_axis = np.ones(self.n_data)      # data value, in this example the value is 1

        # call function inside the class
        self.win, self.label = self.win_rect()    # choose your window here
        self.plot()  # plot

    def win_rect(self):
        return 1 * self.y_axis, "Rectangular"

    def win_tri(self):
        l_ = self.n_data
        return 1 - np.abs((self.x_axis - (self.n_data / 2)) / (l_ / 2)), "Triangular"

    def win_parzen(self):
        l_ = self.n_data + 1
        w0, w = np.zeros(self.n_data), np.zeros(self.n_data)
        for i in range(0, int(l_ / 4)):
            w0[i] = 1 - 6 * ((self.x_axis[i] / (l_ / 2)) ** 2) * (1 - (np.abs(self.x_axis[i]) / (l_ / 2)))
        for i in range(int(l_ / 4), int(l_ / 2)):
            w0[i] = 2 * ((1 - (np.abs(self.x_axis[i]) / (l_ / 2))) ** 3)
        for i in range(int(self.n_data / 2)):
            w[i] = w0[int((self.n_data / 2) - i)]
        for i in range(int(self.n_data / 2), self.n_data):
            w[i] = w0[int(i - (self.n_data / 2))]
        return w, "Parzen"

    def win_welch(self):
        return 1 - ((self.x_axis - (self.n_data / 2)) / (self.n_data / 2)) ** 2, "Welch"

    def win_sine(self):
        return np.sin((np.pi * self.x_axis) / self.n_data), "Sine"

    def win_ham(self):
        a0 = 0.54
        return a0 - ((1 - a0) * np.cos((2 * np.pi * self.x_axis) / self.n_data)), "Hamming"

    def win_han(self):
        a0 = 0.5
        return a0 - ((1 - a0) * np.cos((2 * np.pi * self.x_axis) / self.n_data)), "Hann"

    def win_black(self):
        alpha = 0.16
        return (((1 - alpha) / 2) - 0.5 * np.cos((2 * np.pi * self.x_axis) / self.n_data)
                + (alpha / 2) * np.cos((4 * np.pi * self.x_axis) / self.n_data), "Blackman")

    def win_nutall(self):
        a0, a1, a2, a3 = 0.355768, 0.487396, 0.144232, 0.012604
        return (a0 - a1 * np.cos((2 * np.pi * self.x_axis) / self.n_data)
                + a2 * np.cos((4 * np.pi * self.x_axis) / self.n_data)
                - a3 * np.cos((6 * np.pi * self.x_axis) / self.n_data), "Nutall")

    def win_flattop(self):
        a0, a1, a2, a3, a4 = 0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368
        return (a0 - a1 * np.cos((2 * np.pi * self.x_axis) / self.n_data)
                + a2 * np.cos((4 * np.pi * self.x_axis) / self.n_data)
                - a3 * np.cos((6 * np.pi * self.x_axis) / self.n_data)
                + a4 * np.cos((8 * np.pi * self.x_axis) / self.n_data), "Flat-top")

    def win_gauss(self):
        sigma = 0.3
        return np.exp(-0.5 * ((self.x_axis - (self.n_data / 2)) / (sigma * (self.n_data / 2))) ** 2), "Gauss"

    def win_normal(self):
        sigma, p_ = 0.3, 4
        return np.exp(
            -1 * ((self.x_axis - (self.n_data / 2)) / (sigma * (self.n_data / 2))) ** p_), "Generalized Normal"

    def win_tukey(self):
        alpha = 0.5
        w = np.zeros(self.n_data)
        for i in range(0, int((alpha * self.n_data) / 2)):
            w[i] = 0.5 * (1 - np.cos((2 * np.pi * self.x_axis[i]) / (alpha * self.n_data)))
        for i in range(int((alpha * self.n_data) / 2), int(self.n_data / 2)):
            w[i] = 1
        for i in range(0, int(self.n_data / 2)):
            w[self.n_data - i - 1] = w[i]
        return w, "Tukey"

    def win_plancktaper(self):
        eps = 0.25
        w = np.zeros(self.n_data)
        for i in range(1, int(eps * self.n_data)):
            w[i] = (1 + np.exp(((eps * self.n_data) / self.x_axis[i]) - (
                        (eps * self.n_data) / (eps * self.n_data - self.x_axis[i])))) ** -1
        for i in range(int(eps * self.n_data), int(self.n_data / 2)):
            w[i] = 1
        for i in range(0, int(self.n_data / 2)):
            w[self.n_data - i - 1] = w[i]
        return w, "Planck-taper"

    def win_kaiser(self):
        alpha = 2
        x = np.pi * alpha * np.sqrt(1 - (((2 * self.x_axis) / (self.n_data - 1)) - 1) ** 2)
        i0 = np.zeros(self.n_data)
        for i in range(self.n_data):
            for m in range(25):
                i0[i] += (1 / (math.factorial(m)) ** 2) * ((x[i] / 2) ** (2 * m))
        return i0 / np.max(i0), "Kaiser"

    def win_poisson(self):
        decibels = 60
        tau = (self.n_data * 8.69) / (2 * decibels)
        return np.exp(-1 * np.abs(self.x_axis - (self.n_data / 2)) * (1 / tau)), "Poisson"

    def win_lanczos(self):
        return np.sinc(((2 * self.x_axis) / self.n_data) - 1), "Lanczos"

    def plot(self):
        plt.plot(self.x_axis, self.win, linewidth=2, label=self.label + " Window")
        plt.axhline(0, color='r', linewidth=0.5)
        plt.axhline(1, color='r', linewidth=0.5)
        plt.xlim(self.x_start, len(self.win) - 1)
        plt.ylim(-0.15, 1.15)
        plt.legend(loc='upper right')
        plt.show()


if __name__ == '__main__':
    main_window()
