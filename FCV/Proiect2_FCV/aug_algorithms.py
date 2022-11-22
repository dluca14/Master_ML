import cv2
import numpy as np
from tqdm import tqdm


def rotate(image, param, show=False):
    (height, width, _) = image.shape

    x_radius = param * (np.pi / 180)

    rotated_image_x = np.zeros_like(image)
    centre_x = width / 2
    centre_y = height / 2

    for image_x in tqdm(range(width)):
        for image_y in range(height):
            y_aux = centre_y - image_y
            x_aux = centre_x - image_x

            x_new = x_aux * np.cos(x_radius) + y_aux * np.sin(x_radius)
            y_new = y_aux * np.cos(x_radius) - x_aux * np.sin(x_radius)
            x_new = int(centre_x - x_new)
            y_new = int(centre_y - y_new)

            if 0 < x_new < width and 0 < y_new < height:
                rotated_image_x[y_new][x_new][:] = image[image_y][image_x][:]

    if show:
        final_image = np.hstack((image, rotated_image_x))
        cv2.imshow('Images: before and after', final_image)
        cv2.waitKey()

    return rotated_image_x


def flip(image, param, show=False):
    (height, width, _) = image.shape
    x, y = param[0], param[1]

    if x == 1:
        flipped_image_x = image[:, ::-1, :]
    else:
        flipped_image_x = image

    if y == 1:
        flipped_image_y = flipped_image_x[::-1, :, :]
    else:
        flipped_image_y = flipped_image_x

    if show:
        final_image = np.hstack((image, flipped_image_y))
        cv2.imshow('Images: before and after', final_image)
        cv2.waitKey()
    return flipped_image_y


def translate(image, param, show=True):
    x, y = param[0], param[1]
    translated_image_x = np.zeros_like(image)
    translated_image_y = np.zeros_like(image)

    if x > 0:
        translated_image_x[:, x:, :] = image[:, :-x, :]
    elif x < 0:
        x = abs(x)
        translated_image_x[:, :-x, :] = image[:, x:, :]
    else:
        translated_image_x = image

    if y > 0:
        translated_image_y[y:, :, :] = translated_image_x[:-y, :, :]
    elif y < 0:
        y = abs(y)
        translated_image_y[:-y, :, :] = translated_image_x[y:, :, :]
    else:
        translated_image_y = translated_image_x

    if show:
        final_image = np.hstack((image, translated_image_y))
        cv2.imshow('Images: before and after', final_image)
        cv2.waitKey()

    return translated_image_y


# pixel level
def hue(image, param=10, show=False):
    hue_saturation_value = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hue_saturation_value)

    # to switch off the remainder machanism we verify without going past 255
    mask = (255 - hue) < param
    hue = np.where(mask, 255, hue + param)
    new_image = cv2.merge((hue, saturation, value))
    new_image_bgr = cv2.cvtColor(new_image, cv2.COLOR_HSV2BGR)

    if show:
        final_image = np.hstack((image, new_image_bgr))
        cv2.imshow('Images: before and after', final_image)
        cv2.waitKey()

    return new_image_bgr


def saturation(image, param=10, show=False):
    hue_saturation_value = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hue_saturation_value)

    # to switch off the remainder machanism we verify without going past 255
    mask = (255 - saturation) < param
    saturation = np.where(mask, 255, saturation + param)
    new_image = cv2.merge((hue, saturation, value))
    new_image_bgr = cv2.cvtColor(new_image, cv2.COLOR_HSV2BGR)

    if show:
        final_image = np.hstack((image, new_image_bgr))
        cv2.imshow('Images: before and after', final_image)
        cv2.waitKey()

    return new_image_bgr


def brightness(image, param=10, show=False):
    hue_saturation_value = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hue_saturation_value)

    # to switch off the remainder machanism we verify without going past 255
    if param >= 0:
        mask = (255 - value) < param
        value = np.where(mask, 255, value + param)
    else:
        mask = value < abs(param)
        print(mask)
        value = np.where(mask, 0, value + param)
        print(value)

    new_image = cv2.merge((hue, saturation, value))
    new_image_bgr = cv2.cvtColor(new_image, cv2.COLOR_HSV2BGR)

    if show:
        final_image = np.hstack((image, new_image_bgr))
        cv2.imshow('Images: before and after', final_image)
        cv2.waitKey()

    return new_image_bgr
