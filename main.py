from ImageProcessor import ImageProcessor
from FFTMethod import FFTMethod

import os
import sys


def process(set_dir, plot=False, cc=5, N=None):
    FFTMethod.clear_coefficients()
    fft_method = None
    for image in os.listdir(set_dir):
        if image.lower().endswith('png'):
            path = os.path.join(set_dir, image)
            ip = ImageProcessor()
            ip.read_img(path)
            ip.process_image()
            if plot:
                ip.plot()

            fft_method = FFTMethod(cc)
            fft_method.find_coefficients(ip.result, plot=plot)
    if fft_method:
        fft_method.generate_rankings()
        if plot:
            fft_method.plot_all_coefficients()


def from_bash(dir, N):
    cc = 20
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
        "set0": os.path.join(".", "data", "set0"),
        "set1": os.path.join(".", "data", "set1"),
        "set2": os.path.join(".", "data", "set2"),
        "set3": os.path.join(".", "data", "set3"),
        "set4": os.path.join(".", "data", "set4"),
        "set5": os.path.join(".", "data", "set5"),
        "set6": os.path.join(".", "data", "set6"),
        "set7": os.path.join(".", "data", "set7"),
        "set8": os.path.join(".", "data", "set8"),
    }
    process(test_sets[set_name], cc=cc, plot=plot)


def test():
    path = os.path.join(".", "proj1_daneB", "set7", "19.png")
    ip = ImageProcessor()
    ip.read_img(path)
    ip.process_image()
    ip.plot()
    fft_method = FFTMethod(15)
    fft_method.find_coefficients(ip.result, plot=True)


if __name__ == "__main__":
    # main("set1", 15, False)
    # test()
    if len(sys.argv) != 2:
        exit(1)
    from_bash(sys.argv[0], sys.argv[1])
