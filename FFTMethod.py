import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.signal import cwt


class FFTMethod:
    coefficients = []
    # two list of fft coefficients for each rectangle
    # first - normal curve
    # second - flipped curve

    def __init__(self, coefficients_count, extended_length=1000):
        self.coefficients_count = coefficients_count
        self.extended_length = extended_length

    def find_coefficients(self, rectangle):
        curve = np.argmax(rectangle != 0, axis=0) # find curve
        curve = np.interp(np.linspace(0, self.extended_length - 1, self.extended_length),
                          np.linspace(0, self.extended_length - 1, curve.shape[0]), curve)
        # "rescale" curve to specified width
        flipped_curve = rectangle.shape[0] - np.flip(curve, axis=0)
        # find flipped curve (adjusted to second part of rectangle)
        curve = curve - np.min(curve)
        curve = curve / np.max(curve)

        flipped_curve = flipped_curve - np.min(flipped_curve)
        flipped_curve = flipped_curve / np.max(flipped_curve)
        # normalize curves
        coefficients = [
            np.fft.rfft(curve)[:1 + self.coefficients_count],
            np.fft.rfft(flipped_curve)[:1 + self.coefficients_count],
        ]
        # find coefficients
        self.coefficients.append(coefficients)

    @staticmethod
    def plot(first, second, first_name=None, second_name=None):
        # for debugging purposes

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
        # for debugging purposes
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
                    # find a distance between two normal curve and flipped curve
                    # if the distance is shorter then curves are more fitted
                    values.append(distance.euclidean(self.coefficients[j][0], self.coefficients[i][1]))
                else:
                    values.append(np.inf)
            rank = ""
            for x in np.argsort(np.array(values)):
                # find the a rank by sorting distances
                if i != x:
                    rank += str(x) + " "
            if print_out:
                print(rank.rstrip())
            ranks.append(rank)
        return ranks