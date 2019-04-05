import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.data as data
import skimage.color as color
import skimage
import os
import cv2


def _hough_transform(edges, img):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite('houghlines.jpg', img)


def _process_image(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, 100, 200)
    _hough_transform(edges, img)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()


def main():
    # datasetdir = ".\\proj1_daneA\\set0"
    # for image in os.listdir(datasetdir):
    #     if image.lower().endswith('png'):
    #         print(datasetdir + "\\" + image)
    #         img = io.imread(datasetdir + "\\" + image)
    filedir = ".\\proj1_daneA\\set0\\1.png"
    img = io.imread(filedir)
    _process_image(img)



if __name__ == "__main__":
    main()

