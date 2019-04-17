from ImageProcessor import ImageProcessor
from FFTMethod import FFTMethod

import os
import sys
import matplotlib.pyplot as plt

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


def process(set_dir, image_count, plot=False, write_spec=False, cc=1):
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
        fft_method.find_coefficients(ip.result)

    if fft_method:
        ranks = fft_method.generate_rankings(print_out=plot)
        results = check_ranks(ranks, set_dir, image_count)
        if write_spec:
            print(results)
        if plot:
            fft_method.plot_all_coefficients(only_print=True)
        return results[-1]


def from_bash(dir, N):
    cc = 1
    FFTMethod.clear_coefficients()
    fft_method = None
    for i in range(N):
        path = os.path.join(dir, "{}.png".format(i))
        ip = ImageProcessor()
        ip.read_img(path)
        ip.process_image()
        fft_method = FFTMethod(cc)
        fft_method.find_coefficients(ip.result)
    if fft_method:
        fft_method.generate_rankings()


def main(set_name, cc, plot, write_spec=False):
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
    return process(test_sets[set_name][0], test_sets[set_name][1], cc=cc, plot=plot, write_spec=write_spec)
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
    all_sets_results = 0
    for i in range(9):
        s = "set{}".format(i)
        res = main(s, 1, False, write_spec=True)
        print(s, res)
        all_sets_results += res
    print("All sets res {} is {}%".format(all_sets_results, (all_sets_results / 606) * 100))
    # print("max", max_result, "params", params)
    # test("set1",[2, 5, 15], 20)
    # if len(sys.argv) != 3:
    #     exit(1)
    # from_bash(sys.argv[1], int(sys.argv[2]))
