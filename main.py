from ImageProcessor import ImageProcessor
from FFTMethod import FFTMethod

import os
import sys


def process(set_dir, image_count, plot=False, cc=5, N=None):
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
        fft_method.find_coefficients(ip.result, path.split("\\")[-1][:-4], plot=plot)
    if fft_method:
        fft_method.generate_rankings()
        if plot:
            fft_method.plot_all_coefficients(only_print=True)


def from_bash(dir, N):
    cc = 5
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


def main(set_name, cc, plot):
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
    process(test_sets[set_name][0], test_sets[set_name][1], cc=cc, plot=plot)


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
        fft_method.find_coefficients(ip.result, number, plot=True)
    if fft_method:
        fft_method.generate_rankings()
        fft_method.plot_all_coefficients()


if __name__ == "__main__":
    # main("set1", 8, True)
    # test("set1",[2, 5, 15], 20)
    if len(sys.argv) != 2:
        exit(1)
    from_bash(sys.argv[0], sys.argv[1])
