from ImageProcessor import ImageProcessor
from FFTMethod import FFTMethod

import os


def process(set_dir, plot=False, cc=5):
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


def main(set_name, cc, plot):
    test_sets = {
        "set0": os.path.join(".", "proj1_daneA", "set0"),
        "set1": os.path.join(".", "proj1_daneA", "set1"),
        "set2": os.path.join(".", "proj1_daneA", "set2"),
        "set3": os.path.join(".", "proj1_daneA", "set3"),
        "set4": os.path.join(".", "proj1_daneA", "set4"),
        "set5": os.path.join(".", "proj1_daneA", "set5"),
        "set6": os.path.join(".", "proj1_daneA", "set6"),
        "set7": os.path.join(".", "proj1_daneB", "set7"),
        "set8": os.path.join(".", "proj1_daneB", "set8"),
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
    main("set1", 15, False)
    # test()
