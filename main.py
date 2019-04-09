import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.data as data
import skimage.color as color
import skimage
import os
import cv2
from scipy.spatial import distance

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
    print(angles)
    print(len(angles))
    if len(angles) == 1:
        print("Two lines found")
        if _approx_equal(angles[0][0], 90, 0.05):
            print('Found a probable base')
            probable_indices_of_base.append(angles[0][1])
            probable_indices_of_base.append(angles[0][2])
    else:
        for i in range(0, len(angles)):
            for j in range(i+1, len(angles)):
                    if angles[i][1] == angles[j][1] or angles[i][2] == angles[j][2]:
                        if _approx_equal(angles[i][0] + angles[j][0], 180, 0.05):
                            print('Found a probable base')
                            print(angles[i][0] + angles[j][0])
                            probable_indices_of_base.append(list(set.intersection(set(angles[i]), set(angles[j])))[0])

    print(probable_indices_of_base)
    if len(probable_indices_of_base) is not 0:
        base_found = True
    else:
        probable_indices_of_base.append(0)
    print(angles)
    return base_found, probable_indices_of_base


def _hough_transform(edges, img):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, 5, 20, 20)  # czwarty parametr to minimalna długość linii
    declinations = []
    if lines is not None:
        if len(lines) > 1:
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
        else:
            for i in range(0, len(lines)):
                line = lines[i][0]
                cv2.line(img, (line[0], line[1]), (line[2], line[3]), (110, 40, 255), 3, cv2.LINE_AA)
    # print(lines)
    return declinations, lines


def _process_image(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, 100, 200)
    declinations, lines = _hough_transform(edges, img)
    print(lines)
    if lines is not None:
        if len(lines) > 1:
            found, index = _check_angles(declinations)
            if found:
                if len(index) == 1:
                    line = lines[index][0][0]
                    cv2.line(edges, (line[0], line[1]), (line[2], line[3]), (110, 40, 255), 3, cv2.LINE_AA)
                if len(index) > 1:
                    print("Several possible bases")
                    line_lengths = []
                    for i in index:
                        line = lines[i][0]
                        p1 = (int(line[0]), int(line[1]))
                        p2 = (int(line[2]), int(line[3]))
                        print(p1)
                        dst = distance.euclidean(p1, p2)
                        line_lengths.append(dst)
                        print(line_lengths)
                    max_line_index = line_lengths.index(max(line_lengths))
                    line = lines[max_line_index][0]
                    cv2.line(edges, (line[0], line[1]), (line[2], line[3]), (110, 40, 255), 3, cv2.LINE_AA)
        else:
            print("One line found")
            line = lines[0][0]
            cv2.line(edges, (line[0], line[1]), (line[2], line[3]), (110, 40, 255), 3, cv2.LINE_AA)
    else:
        print("Lines not found")
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()


def main():
    datasetdir = os.path.join(".", "proj1_daneA", "set1")
    for image in os.listdir(datasetdir):
        if image.lower().endswith('png'):
            path = os.path.join(datasetdir, image)
            print(path)
            img = io.imread(path)
            _process_image(img)
    # filedir = ".\\proj1_daneA\\set0\\1.png"
    # img = io.imread(filedir)
    # _process_image(img)


if __name__ == "__main__":
    main()

