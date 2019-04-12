from time import sleep

import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.data as data
import skimage.color as color
import skimage.measure
import skimage
import os
import cv2


def get_sub_image(rect, src):
    # From stackoverflow
    # Get center, size, and angle from rect
    center, size, theta = rect
    # Convert to int
    center, size = tuple(map(int, center)), tuple(map(int, size))
    # Get rotation matrix for rectangle
    M = cv2.getRotationMatrix2D( center, theta, 1)
    # Perform rotation on src image
    dst = cv2.warpAffine(src, M, src.shape[:2])
    out = cv2.getRectSubPix(dst, size, center)
    return out


def get_four_parts_indices(img_shape):
    # It returns sets of ranges (each for one corner)
    rows, cols = img_shape
    rows_cond = rows % 2
    cols_cond = cols % 2
    results = []
    center_x = int(cols / 2)
    center_y = int(rows / 2)
    if rows_cond and cols_cond: #
        center_y = int(rows / 2) + 1
        center_x = int(cols / 2) + 1
    elif rows_cond: # only y is odd
        center_y = int(rows / 2) + 1
    elif cols_cond: # only x is odd
        center_x = int(cols / 2) + 1
    results.append(((0, 0), (center_y, center_x)))
    results.append(((center_y, 0), (rows, center_x)))
    results.append(((center_y, center_x), (rows, cols)))
    results.append(((0, center_x), (center_y, cols)))
    return results


def find_base(img_shape, areas):
    # It finds a base of the rectangle, chooses maximum sum from two adjacent rectangles
    rows, cols = img_shape
    areas_sum = []
    for i, _ in enumerate(areas):
        if i != len(areas) - 1:
            areas_sum.append(areas[i] + areas[i + 1])
        else:
            areas_sum.append(areas[0] + areas[i])
    arg_max_sum = np.argmax(areas_sum)
    if arg_max_sum == 0:
        return (0, 0), (rows, 0)
    if arg_max_sum == 1:
        return (rows, 0), (rows, cols)
    if arg_max_sum == 2:
        return (0, cols), (rows, cols)
    else:
        return (0, 0), (0, cols)


def find_best_rectangle(contours):
    # It finds the best rectangle from contour If there is more than one contour, it chooses the biggest
    best_rect = None
    off_1, off_2 = -1, -1
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        if rect[1][0] > off_1 and rect[1][1] > off_2:
            best_rect = rect
            off_1 = rect[1][0]
            off_2 = rect[1][1]
    return best_rect


def process_image(img):
    # scale_y = img.shape[0] / 100
    # scale_x = img.shape[1] / 100
    # print(scale_y, scale_x, img.shape)
    # img = cv2.resize(img, (int(img.shape[0] / scale_y), int(img.shape[1] / scale_x)))
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    _, contours, hierarchy = cv2.findContours(thresh, 1, 2)
    if len(contours) > 1:
        rect = find_best_rectangle(contours)
    else:
        rect = cv2.minAreaRect(contours[0])
    dst = get_sub_image(rect, thresh)
    plt.show()
    ranges = get_four_parts_indices(dst.shape[:2])
    areas = []
    for range in ranges:
        lower = range[0]
        upper = range[1]
        rectangle_part = dst[lower[0]:upper[0], lower[1]: upper[1]]
        arguments = np.argwhere(rectangle_part > 127)
        areas.append(arguments.shape[0])
    base = find_base(dst.shape[:2], areas)
    # for point in rect:
    #     print(point)
    #     x, y = int(point[0]), int(point[1])
    #     cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(img, [box], 0, (255, 0, 0), 1)
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2RGB)
    cv2.line(dst, (base[0][1], base[0][0]), (base[1][1], base[1][0]), (255, 0, 0), 3)
    return img, dst

def plot(img, res, name):
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(img)

    plt.subplot(122)
    plt.imshow(res)
    plt.title(name)
    plt.show()


def main():
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
    set_dir = test_sets["set2"]

    for image in os.listdir(set_dir):
        if image.lower().endswith('png'):
            path = os.path.join(set_dir, image)
            img = io.imread(path)
            img, res = process_image(img)
            plot(img, res, path.split("\\")[-1])


def test():
    path = os.path.join(".", "proj1_daneA", "set2", "7.png")
    img = io.imread(path)
    img, res = process_image(img)
    plot(img, res, "test")


if __name__ == "__main__":
    main()
    # test()
