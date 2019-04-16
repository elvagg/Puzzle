from ImageProcessor import ImageProcessor
from FFTMethod import FFTMethod

import os


def main(set_name):
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
    set_dir = test_sets[set_name]

    for image in os.listdir(set_dir):
        if image.lower().endswith('png'):
            path = os.path.join(set_dir, image)
            ip = ImageProcessor()
            ip.read_img(path)
            ip.process_image()
            ip.plot()


def test():
    path = os.path.join(".", "proj1_daneA", "set2", "7.png")
    ip = ImageProcessor()
    ip.read_img(path)
    ip.process_image()
    ip.plot()


if __name__ == "__main__":
    main("set0")
    # test()
