from ImageProcessor import ImageProcessor
from FFTMethod import FFTMethod

import os
import sys

import numpy as np


def check_ranks(ranks, path, N):
    with open(os.path.join(path, 'correct.txt'), 'rt') as f:
        correct = f.read().splitlines()
    size = 6
    result = np.zeros(size)

    if len(ranks) != len(correct):
        return result

    for line, c in zip(ranks, correct):
        line = line.split()
        try:
            idx = line.index(c)
        except:
            idx = N - 1
        if idx < size - 1:
            result[idx] += 1
        result[-1] += 1.0 / (1.0 + idx)
    return result


def process(set_dir, image_count, plot=False, cc=5, N=None, method_fft_coeffs=None, extend=None):
    FFTMethod.clear_coefficients()
    fft_method = None
    for i in range(image_count):
        path = os.path.join(set_dir, "{}.png".format(i))
        ip = ImageProcessor()
        ip.read_img(path)
        ip.process_image()
        if plot:
            ip.plot()

        fft_method = FFTMethod(cc)
        fft_method.find_coefficients(ip.result,plot=plot, method_fft_coeffs=method_fft_coeffs, extend=extend)
    if fft_method:
        ranks = fft_method.generate_rankings(print_out=False)
        results = check_ranks(ranks, set_dir, image_count)
        # print("results", results[:-1], "rank:", results[-1])
        return results[-1]
        # if plot:
        #     fft_method.plot_all_coefficients(only_print=True)


def from_bash(dir, N):
    cc = 17
    FFTMethod.clear_coefficients()
    fft_method = None
    for i in range(N):
        path = os.path.join(dir, "{}.png".format(i))
        ip = ImageProcessor()
        ip.read_img(path)
        ip.process_image()
        fft_method = FFTMethod(cc)
        fft_method.find_coefficients(ip.result, plot=False)
    if fft_method:
        fft_method.generate_rankings()


def main(set_name, cc, plot, method_fft_coeffs=None, extend=None):
    test_sets = {
        "set0": (os.path.join(".", "data", "set0"), 6),
        "set1": (os.path.join(".", "data", "set1"), 20),
        "set2": (os.path.join(".", "data", "set2"), 20),
        "set3": (os.path.join(".", "data", "set3"), 20),
        "set4": (os.path.join(".", "data", "set4"), 20),
        "set5": (os.path.join(".", "data", "set5"), 200),
        "set6": (os.path.join(".", "data", "set6"), 200),
        "set7": (os.path.join(".", "data", "set7"), 20),
        "set8": (os.path.join(".", "data", "set8"), 100),
    }
    return process(test_sets[set_name][0], test_sets[set_name][1], cc=cc, plot=plot,
                   method_fft_coeffs=method_fft_coeffs, extend=extend)
    # from_bash(test_sets[set_name][0], test_sets[set_name][1])


def test(data_set, numbers, cc):
    fft_method = None
    FFTMethod.clear_coefficients()
    for number in numbers:
        path = os.path.join(".", "data", data_set, "{}.png".format(number))
        ip = ImageProcessor()
        ip.read_img(path)
        ip.process_image()
        ip.plot()
        fft_method = FFTMethod(cc)
        fft_method.find_coefficients(ip.result, plot=True)
    if fft_method:
        fft_method.generate_rankings()
        fft_method.plot_all_coefficients()


if __name__ == "__main__":
    max_result = -1
    params = -1
    for ex in range(2):
        for h in range(4):
            for i in range(1, 250):
                all_sets_result = 0
                for j in range(9):
                    all_sets_result += main("set{}".format(j), i, False, method_fft_coeffs=h, extend=bool(ex))
                if all_sets_result > max_result:
                    max_result = all_sets_result
                    params = (bool(ex), h, i)
                print('{} {} {} result {}'.format(bool(ex), h, i, all_sets_result), '--- best ---' + '{} result {}'.format(params, max_result))
    print("max", max_result, "params", params)
    # test("set1",[2, 5, 15], 20)
    # if len(sys.argv) != 3:
    #     exit(1)
    # from_bash(sys.argv[1], int(sys.argv[2]))
