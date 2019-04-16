import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.data as data
import skimage.color as color
import skimage.measure
import skimage
import cv2


class ImageProcessor:
    def __init__(self):
        self.img = None
        # loaded image
        self.result = None
        # cut and rotated rectangle
        self.name = ""

    def get_sub_image(self, rect, src):
        center, size, theta = rect
        center, size = tuple(map(int, center)), tuple(map(int, size))
        (h, w) = src.shape[:2]
        (cX, cY) = center
        M, nW, nH = self.get_rotation_matrix(cX, cY, theta, h, w)
        dst = cv2.warpAffine(src, M, (nW, nH))
        out = cv2.getRectSubPix(dst, size, (nW/2, nH/2))
        return out

    def get_final_rotation(self, image, base):
        angle = 0
        (h, w) = image.shape[:2]
        (cX, cY) = (w//2, h//2)
        if base[0][0] == 0 and base[1][1] == 0:
            angle = 90
        elif base[0][0] == 0 and base[1][0] == 0:
            angle = 180
        elif base[0][1] != 0:
            angle = -90
        M, nW, nH = self.get_rotation_matrix(cX, cY, angle, h, w)
        out = cv2.warpAffine(image, M, (nW, nH))
        return out

    @staticmethod
    def get_four_parts_indices(img_shape):
        # It returns sets of ranges (each for one corner)
        rows, cols = img_shape
        rows_cond = rows % 2
        cols_cond = cols % 2
        results = []
        center_x = int(cols / 2)
        center_y = int(rows / 2)
        if rows_cond and cols_cond:  #
            center_y = int(rows / 2) + 1
            center_x = int(cols / 2) + 1
        elif rows_cond:  # only y is odd
            center_y = int(rows / 2) + 1
        elif cols_cond:  # only x is odd
            center_x = int(cols / 2) + 1
        results.append(((0, 0), (center_y, center_x)))
        results.append(((center_y, 0), (rows, center_x)))
        results.append(((center_y, center_x), (rows, cols)))
        results.append(((0, center_x), (center_y, cols)))
        return results

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def get_rotation_matrix(cX, cY, angle, h, w):
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        return M, nW, nH

    def process_image(self):
        ret, thresh = cv2.threshold(self.img, 127, 255, cv2.THRESH_BINARY)
        _, contours, hierarchy = cv2.findContours(thresh, 1, 2)
        if len(contours) > 1:
            rect = self.find_best_rectangle(contours)
        else:
            rect = cv2.minAreaRect(contours[0])
        self.result = self.get_sub_image(rect, thresh)

        ranges = self.get_four_parts_indices(self.result.shape[:2])
        areas = []

        for range in ranges:
            lower = range[0]
            upper = range[1]
            rectangle_part = self.result[lower[0]:upper[0], lower[1]: upper[1]]
            arguments = np.argwhere(rectangle_part > 127)
            areas.append(arguments.shape[0])

        base = self.find_base(self.result.shape[:2], areas)
        self.result = self.get_final_rotation(self.result, base)

    def read_img(self, path):
        self.img = io.imread(path)
        self.name = path.split("\\")[-1]

    def plot(self):
        plt.figure(1)
        plt.subplot(121)
        plt.imshow(self.img)

        plt.subplot(122)
        plt.imshow(self.result)
        plt.title(self.name)
        plt.show()

    @staticmethod
    def scale_image(img, rect_width, rect_height):
        return cv2.resize(img, (rect_width, rect_height), interpolation=cv2.INTER_CUBIC)

