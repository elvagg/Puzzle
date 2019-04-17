import numpy as np
import matplotlib.pyplot as plt
import sys

class FFTMethod:
    coefficients = []

    # two list of fft coefficients for each rectangle
    # first - normal curve
    # second - flipped curve

    def __init__(self, coefficients_count):
        self.coefficients_count = coefficients_count
        self.extended_length = 500

    def find_coefficients(self, rectangle, plot=False, method_fft_coeffs=0, extend=True):
        curve = np.argmax(rectangle != 0, axis=0)
        if extend:
            curve = np.interp(np.linspace(0, self.extended_length - 1, self.extended_length),
                              np.linspace(0, self.extended_length - 1, curve.shape[0]), curve)
        flipped_curve = rectangle.shape[0] - np.flip(curve, axis=0)
        curve = curve - np.min(curve)
        curve = curve / np.max(curve)

        flipped_curve = flipped_curve - np.min(flipped_curve)
        flipped_curve = flipped_curve / np.max(flipped_curve)

        if plot:
            self.plot(curve, flipped_curve)

        if method_fft_coeffs == 0:
            coefficients = [
                np.fft.fft(curve, n=self.coefficients_count + 1)[1:int(self.coefficients_count / 2) + 1],
                np.fft.fft(flipped_curve, n=self.coefficients_count + 1)[1:int(self.coefficients_count / 2) + 1],
            ]
        elif method_fft_coeffs == 1:
            coefficients = [
                np.fft.fft(curve)[1:1 + self.coefficients_count],
                np.fft.fft(flipped_curve)[1:1 + self.coefficients_count],
            ]
        elif method_fft_coeffs == 2:
            coefficients = [
                np.fft.fft(curve, n=self.coefficients_count + 1)[0:int(self.coefficients_count / 2) + 1],
                np.fft.fft(flipped_curve, n=self.coefficients_count + 1)[0:int(self.coefficients_count / 2) + 1],
            ]
        else:
            coefficients = [
                np.fft.fft(curve)[0: self.coefficients_count],
                np.fft.fft(flipped_curve)[0: self.coefficients_count],
            ]
        self.coefficients.append(coefficients)

    @staticmethod
    def plot(first, second, first_name=None, second_name=None):
        plt.figure(1)
        ax = plt.subplot(121)
        ax.plot(first)
        if first_name:
            ax.set_title(first_name)

        ax = plt.subplot(122)
        ax.plot(second)
        if second_name:
            ax.set_title(second_name)
        plt.show()

    @classmethod
    def clear_coefficients(cls):
        cls.coefficients = []

    def plot_all_coefficients(self, only_print=False):
        for i in range(len(self.coefficients)):
            for j in range(len(self.coefficients)):
                if not only_print:
                    self.plot(np.abs(self.coefficients[i][0]),
                              np.abs(self.coefficients[j][1]),
                              first_name=str(i), second_name=str(j))
                print("{} {} - {}".format(i, j, np.abs(np.average(self.coefficients[i][0] -
                                                                  self.coefficients[j][1]))))
            print("-" * 40)

    def generate_rankings(self, print_out=True):
        ranks = []
        for i in range(len(self.coefficients)):
            values = []
            for j in range(len(self.coefficients)):
                if i != j:
                    values.append(np.abs(np.average(self.coefficients[i][0] - self.coefficients[j][1])))
                else:
                    values.append(np.inf)
            rank = ""
            for x in np.argsort(values):
                if i != x:
                    rank += str(x) + " "
            if print_out:
                print(rank.rstrip())
            ranks.append(rank)
        return ranks
