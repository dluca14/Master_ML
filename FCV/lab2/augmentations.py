from PIL.Image import new
import cv2
import numpy as np
import math
from scipy.signal import convolve2d


def brightness(image, p=["10"], show=False):
    if type(p) == list:
        v = int(p[0]) if p[0].isnumeric() else 10
    else:
        v = 10

    # print(v)
    hue_saturation_value = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hue_saturation_value)

    # to switch off the remainder machanism we verify without going past 255
    if v >= 0:
        mask = (255 - value) < v
        value = np.where(mask, 255, value + v)
    else:
        mask = value < abs(v)
        print(mask)
        value = np.where(mask, 0, value + v)
        print(value)

    new_image = cv2.merge((hue, saturation, value))
    new_image_bgr = cv2.cvtColor(new_image, cv2.COLOR_HSV2BGR)

    if show:
        final_image = np.hstack((image, new_image_bgr))
        cv2.imshow('Images: before and after', final_image)
        cv2.waitKey()

    return new_image_bgr


def hue(image, p=["10"], show=False):
    if type(p) == list:
        v = int(p[0]) if p[0].isnumeric() else 10
    else:
        v = 10
    # print(v)
    hue_saturation_value = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hue_saturation_value)

    # to switch off the remainder machanism we verify without going past 255
    mask = (255 - hue) < v
    hue = np.where(mask, 255, hue + v)
    new_image = cv2.merge((hue, saturation, value))
    new_image_bgr = cv2.cvtColor(new_image, cv2.COLOR_HSV2BGR)

    if show:
        final_image = np.hstack((image, new_image_bgr))
        cv2.imshow('Images: before and after', final_image)
        cv2.waitKey()

    return new_image_bgr


def saturation(image, p=["10"], show=False):
    if type(p) == list:
        v = int(p[0]) if p[0].isnumeric() else 10
    else:
        v = 10
    # print(v)
    hue_saturation_value = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hue_saturation_value)

    # to switch off the remainder machanism we verify without going past 255
    mask = (255 - saturation) < v
    saturation = np.where(mask, 255, saturation + v)
    new_image = cv2.merge((hue, saturation, value))
    new_image_bgr = cv2.cvtColor(new_image, cv2.COLOR_HSV2BGR)

    if show:
        final_image = np.hstack((image, new_image_bgr))
        cv2.imshow('Images: before and after', final_image)
        cv2.waitKey()

    return new_image_bgr


def sigmoid(x, alpha):
    return 255 / (1 + math.exp(-alpha * (x - 127) / 32))


def inverseMapping(x):
    return 255 - x


def toneMappingInverse(image, show=False):
    newImage = np.copy(image)
    newImage = newImage.transpose(2, 0, 1)

    inverse_vectorized = np.vectorize(inverseMapping)

    newImage[0] = inverse_vectorized(newImage[0])
    newImage[1] = inverse_vectorized(newImage[1])
    newImage[2] = inverse_vectorized(newImage[2])

    newImage = newImage.transpose(1, 2, 0)

    if show:
        final_image = np.hstack((image, newImage))
        cv2.imshow('Images: before and after', final_image)
        cv2.waitKey()

    return newImage


def toneMappingSigmoid(image, p=["2"], show=False):
    if type(p) == list:
        alpha = int(p[0]) if p[0].isnumeric() else 2
    else:
        alpha = 2
    # print(alpha)
    newImage = np.copy(image)
    newImage = newImage.transpose(2, 0, 1)

    sigmoid_vectorized = np.vectorize(sigmoid)

    newImage[0] = sigmoid_vectorized(newImage[0], alpha)
    newImage[1] = sigmoid_vectorized(newImage[1], alpha)
    newImage[2] = sigmoid_vectorized(newImage[2], alpha)

    newImage = newImage.transpose(1, 2, 0)

    if show:
        final_image = np.hstack((image, newImage))
        cv2.imshow('Images: before and after', final_image)
        cv2.waitKey()

    return newImage


def convolve(image, kernel):
    return convolve2d(image, kernel, 'same', boundary='fill', fillvalue=0)


def createKernel(kernel=(3, 3), sigma=1):
    m, n = [(ss - 1.) / 2. for ss in kernel]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def gaussianBlur(image, p=[7, 7, 8], show=False):
    if type(p) == list and len(p) == 3:
        k1 = int(p[0]) if p[0].isnumeric() else 7
        k2 = int(p[1]) if p[1].isnumeric() else 7
        sigma = int(p[2]) if p[2].isnumeric() else 8
        kernel = (k1, k2)
    else:
        kernel = (7, 7)
        sigma = 8
    # print(kernel,sigma)

    gaussian = createKernel(kernel, sigma)
    newImage = np.copy(image)
    newImage = newImage.transpose(2, 0, 1)

    newImage[0] = convolve(newImage[0], gaussian)
    newImage[1] = convolve(newImage[1], gaussian)
    newImage[2] = convolve(newImage[2], gaussian)

    newImage = newImage.transpose(1, 2, 0)

    if show:
        final_image = np.hstack((image, newImage))
        cv2.imshow('Images: before and after', final_image)
        cv2.waitKey()

    return newImage


def sharpen(image, show=False):
    sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    newImage = np.copy(image)
    newImage = newImage.transpose(2, 0, 1)

    newImage[0] = convolve(newImage[0], sharpen)
    newImage[1] = convolve(newImage[1], sharpen)
    newImage[2] = convolve(newImage[2], sharpen)

    newImage = newImage.transpose(1, 2, 0)

    if show:
        final_image = np.hstack((image, newImage))
        cv2.imshow('Images: before and after', final_image)
        cv2.waitKey()

    return newImage


def histogramEqualizationHelper(image):
    flattened_image = image.flatten()

    histogram = np.bincount(flattened_image, minlength=256)
    normalised_histogram = histogram / np.sum(histogram)
    cumulative_normalised_histogram = np.cumsum(normalised_histogram)

    mapping = np.floor(255 * cumulative_normalised_histogram)
    new_pixel_values = [mapping[p] for p in flattened_image]

    return np.array(new_pixel_values).reshape(image.shape)


def histogramEqualization(image, show=False):
    newImage = np.copy(image)
    blue = newImage[:, :, 0]
    green = newImage[:, :, 1]
    red = newImage[:, :, 2]

    new_blue = histogramEqualizationHelper(blue)
    new_green = histogramEqualizationHelper(green)
    new_red = histogramEqualizationHelper(red)

    newImage[:, :, 0] = new_blue
    newImage[:, :, 1] = new_green
    newImage[:, :, 2] = new_red

    if show:
        final_image = np.hstack((image, newImage))
        cv2.imshow('Images: before and after', final_image)
        cv2.waitKey()

    return newImage

# if __name__ == '__main__':
# image = cv2.imread('../Images/img_12.jpg')
# brightness(image, p=[100],  show=True)
# hue(image, p=[1],  show=True)
# saturation(image, p=[15],  show=True)
# toneMappingInverse(image, show=True)
# toneMappingSigmoid(image, p=[2], show=True)
# gaussianBlur(image, p = [ (7, 7), 8 ], show=True)
# sharpen(image,show=True)
# histogramEqualization(image, show=True)
