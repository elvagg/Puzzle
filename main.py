import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.data as data
import skimage.color as color
import skimage
import os
import cv2


def _approx_equal(x, y, tolerance=0.01):
    return abs(x-y) <= 0.5 * tolerance * (x + y)


def _get_angle_declination(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    rads = np.math.atan2(-dy, dx)
    return np.math.degrees(rads)


def _check_angles(declinations):
    angles = []
    print(declinations)
    base_found = False
    if len(declinations) not in {0, 1}:
        for i in range(0, len(declinations)):
            for j in range(i+1, len(declinations)):
                angles.append((abs(declinations[i] - declinations[j]), i, j))
    elif len(declinations) == 1:
        angles.append(0)

    probable_indices_of_base = []
    print(len(angles))
    if len(angles) == 0:
        print("Lines not found")
    if len(angles) == 1:
        print("One line found")
        probable_indices_of_base.append(list({0}))
    elif len(angles) == 2:
        print("Two lines found")
        if _approx_equal(angles[0][0] + angles[1][0], 90, 0.05):
            print('Found a probable base')
            print(angles[0][0] + angles[1][0])
            probable_indices_of_base.append(list(set.intersection(set(angles[0]), set(angles[1]))))
    else:
        for i in range(0, len(angles)):
            for j in range(i+1, len(angles)):
                    if angles[i][1] == angles[j][1] or angles[i][2] == angles[j][2]:
                        if _approx_equal(angles[i][0] + angles[j][0], 180, 0.05):
                            print('Found a probable base')
                            print(angles[i][0] + angles[j][0])
                            probable_indices_of_base.append(list(set.intersection(set(angles[i]), set(angles[j]))))

    if len(probable_indices_of_base) == 1:
        base_found = True
    else:
        probable_indices_of_base.append(0)
    print(angles)
    return base_found, probable_indices_of_base[0]


def _hough_transform(edges, img):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, 5, 20, 20)  # czwarty parametr to minimalna długość linii
    declinations = []
    if lines is not None:
        for i in range(0, len(lines)):
            # Standardowa tranformacja Hougha cv2.HoughLines(...)
            # rho = lines[i][0][0]
            # theta = lines[i][0][1]
            # a = np.math.cos(theta)
            # b = np.math.sin(theta)
            # x0 = a * rho
            # y0 = b * rho
            # pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            # pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            # cv2.line(edges, pt1, pt2, (110, 40, 255), 3, cv2.LINE_AA)

            # Probabilistyczna transformacja Hougha cv2.HoughLinesP(...)
            line = lines[i][0]
            declinations.append(_get_angle_declination((line[0], line[1]), (line[2], line[3])))
            cv2.line(img, (line[0], line[1]), (line[2], line[3]), (110, 40, 255), 3, cv2.LINE_AA)
    # print(lines)
    return declinations, lines


def _process_image(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, 100, 200)
    declinations, lines = _hough_transform(edges, img)
    found, index = _check_angles(declinations)
    if found:
        line = lines[index][0][0]
        cv2.line(edges, (line[0], line[1]), (line[2], line[3]), (110, 40, 255), 3, cv2.LINE_AA)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()


def main():
    datasetdir = ".\\proj1_daneA\\set1"
    for image in os.listdir(datasetdir):
        if image.lower().endswith('png'):
            print(datasetdir + "\\" + image)
            img = io.imread(datasetdir + "\\" + image)
            _process_image(img)
    # filedir = ".\\proj1_daneA\\set0\\1.png"
    # img = io.imread(filedir)
    # _process_image(img)


if __name__ == "__main__":
    main()

