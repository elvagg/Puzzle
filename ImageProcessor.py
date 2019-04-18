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
        cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        best_rect = cv2.minAreaRect(cnt)
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

    def find_sampled_edge(img, final_image_size):
        # sampling
        edge = []
        sample_parameter = 5
        img = img[:, ::sample_parameter]
        for x in range(int(final_image_size / sample_parameter)):
            curve_height = 0
            for y in range(final_image_size):
                if img[y, x] > 0:
                    curve_height = y
                    break
            edge.append(curve_height)
        return edge

    def four_point_transform(self, image, pts):
        # print(pts)
        rect = np.array(pts, np.float32)
        (tl, tr, br, bl) = rect

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        # return the warped image
        return warped

    def fix_perspective(self, contours):
        warped = self.four_point_transform(self.result, contours)
        plt.plot(warped)
        return warped

    def find_sampled_edge(self, img, final_image_size):
        # sampling
        edge = []
        sample_parameter = 5
        img = img[:, ::sample_parameter]
        for x in range(int(final_image_size / sample_parameter)):
            curve_height = 0
            for y in range(final_image_size):
                if img[y, x] > 0:
                    curve_height = y
                    break
            edge.append(curve_height)
        return edge

    def line_intersection(self, line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])  # Typo was here

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

    def find_four_corners(self, points):
        # TO DO - PÓKI CO ZNAJDUJE ZŁE, TRZEBA ZAIMPLEMENTOWAĆ INACZEJ
        new_points = []
        for p in points:
            new_points.append(list(p[0]))
        new_points.sort(key=lambda x: x[0])
        two_min_x = new_points[:2]
        two_max_x = new_points[-2:]
        y_min = np.min(new_points, axis=0)[1]
        y_max = np.max(new_points, axis=0)[1]
        intersection_points = []
        if (len(two_min_x) > 1 and len(two_max_x) > 1):
            intersection_points.append(self.line_intersection(two_min_x, ((0, y_min), (self.result.shape[1], y_min))))
            intersection_points.append(self.line_intersection(two_max_x, ((0, y_min), (self.result.shape[1], y_min))))
            intersection_points.append(self.line_intersection(two_max_x, ((0, y_max), (self.result.shape[1], y_max))))
            intersection_points.append(self.line_intersection(two_min_x, ((0, y_max), (self.result.shape[1], y_max))))
        else:
            intersection_points.append((0, 0))
            intersection_points.append((self.img.shape[0], 0))
            intersection_points.append((self.img.shape[0], self.img.shape[1]))
            intersection_points.append((0, self.img.shape[1]))
        return intersection_points

    def process_image(self, fix_perspective=False):
        ret, thresh = cv2.threshold(self.img, 230, 255, cv2.THRESH_BINARY)
        _, contours, hierarchy = cv2.findContours(thresh, 1, cv2.CHAIN_APPROX_NONE)

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

        if fix_perspective:
            _, contours, hierarchy = cv2.findContours(self.result, 1, cv2.CHAIN_APPROX_NONE)
            cnt = contours[0]
            # chull = cv2.convexHull(cnt)
            # epsilon = 0.01 * cv2.arcLength(chull, True)
            approx = cv2.approxPolyDP(cnt, 4, True)

            intersection_points = self.find_four_corners(approx)

            if len(intersection_points) > 0:
                x = []
                y = []
                for point in approx:
                    x.append(point[0][0])
                    y.append(point[0][1])
                # plt.figure(1)
                # plt.scatter(x, y)
                # plt.show()
                self.result = self.fix_perspective(intersection_points)
                # self.plot()

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

