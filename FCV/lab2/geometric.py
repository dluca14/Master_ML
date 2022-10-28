import numpy as np
import cv2
from tqdm import tqdm


def read_geometric_parameters(image, p=["100", "100"]):
    (height, width, _) = image.shape

    sign_x = True
    sign_y = True
    if type(p) == list:
        if "-" in p[0]:
            sign_x = False
        if "-" in p[1]:
            sign_y = False

        x = int(p[0].strip('-')) if p[0].strip('-').isnumeric() else 100
        y = int(p[1].strip('-')) if p[1].strip('-').isnumeric() else 100

        x = -x if sign_x == False else x
        y = -y if sign_y == False else y

    else:
        x = 100
        y = 100

    if abs(x) > width:
        x = width
    if abs(y) > height:
        y = height

    return x, y


def translate(image, p=["10", "20"], show=False):
    x, y = read_geometric_parameters(image, p)
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


def flip(image, p=["1", "1"], show=False):
    (height, width, _) = image.shape
    if type(p) == list:
        x = int(p[0]) if p[0].isnumeric() else 1
        y = int(p[1]) if p[1].isnumeric() else 1

    else:
        x = 1
        y = 1
    # print(x,y)
    flipped_image_x = np.zeros_like(image)
    flipped_image_y = np.zeros_like(image)

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


def rotate(image, p=["-10"], show=False):
    (height, width, _) = image.shape
    sign_x = False
    if type(p) == list:
        if "-" in p[0]:
            sign_x = True
        x = int(p[0].strip('-')) if p[0].strip('-').isnumeric() else 30
        x = -x if sign_x == False else x
    else:
        x = 30

    x_radius = x * (np.pi / 180)
    # print(x_radius)

    rotated_image_x = np.zeros_like(image)
    original_centre_x = width / 2
    original_centre_y = height / 2
    rotated_centre_x = width / 2
    rotated_centre_y = height / 2

    for image_x in tqdm(range(width)):
        for image_y in range(height):
            y_aux = original_centre_y - image_y
            x_aux = original_centre_x - image_x

            x_new = x_aux * np.cos(x_radius) + y_aux * np.sin(x_radius)
            y_new = y_aux * np.cos(x_radius) - x_aux * np.sin(x_radius)

            x_new = int(rotated_centre_x - x_new)
            y_new = int(rotated_centre_y - y_new)

            if 0 < x_new < width and 0 < y_new < height:
                rotated_image_x[y_new][x_new][:] = image[image_y][image_x][:]

    if show:
        final_image = np.hstack((image, rotated_image_x))
        cv2.imshow('Images: before and after', final_image)
        cv2.waitKey()
    return rotated_image_x


def scale(image, p=["0.5", "0.75"], show=False):
    (height, width, _) = image.shape

    if type(p) == list:
        x = float(p[0])
        y = float(p[1])
        # print(x,y)
        if x <= 0:
            x = 0.5
        if y <= 0:
            y = 0.75
        # print(x,y)
    else:
        x = 0.5
        y = 0.75

    scaled_image_x = np.zeros_like(image)
    original_centre_x = width / 2
    original_centre_y = height / 2
    scaled_centre_x = width / 2
    scaled_centre_y = height / 2

    scaled_image_x = np.zeros_like(image)
    # resized_scaled_image_x  = np.zeros([int(new_image_height), int(new_image_width),3])

    for image_x in tqdm(range(width)):
        for image_y in range(height):
            y_aux = original_centre_y - image_y
            x_aux = original_centre_x - image_x

            x_new = x_aux * x
            y_new = y_aux * y

            x_new = int(scaled_centre_x - x_new)
            y_new = int(scaled_centre_y - y_new)

            if 0 < x_new < width and 0 < y_new < height:
                scaled_image_x[y_new][x_new][:] = image[image_y][image_x][:]

    # resized_scaled_image_x = scaled_image_x[
    #     int(scaled_centre_y) - new_image_height // 2 : int(scaled_centre_y) + new_image_height // 2,
    #     int(scaled_centre_x) - new_image_width // 2 : int(scaled_centre_x) + new_image_width // 2,:]
    # print(resized_scaled_image_x.shape)

    if show:
        final_image = np.hstack((image, scaled_image_x))
        cv2.imshow('Images: before and after', final_image)
        cv2.waitKey()
    return scaled_image_x


def shear(image, p=["0.5"], show=False):
    (height, width, _) = image.shape
    sign_x = False
    if type(p) == list:
        if "-" in p[0]:
            sign_x = True
        beta = float(p[0].strip('-'))
        beta = -beta if sign_x == False else beta
    else:
        beta = 0.5

    sheared_image_x = np.zeros_like(image)
    original_centre_x = width / 2
    original_centre_y = height / 2
    sheared_centre_x = width / 2
    sheared_centre_y = height / 2

    sheared_image_x = np.zeros_like(image)
    # resized_sheared_image_x  = np.zeros([int(new_image_height), int(new_image_width),3])

    for image_x in tqdm(range(width)):
        for image_y in range(height):
            y_aux = original_centre_y - image_y
            x_aux = original_centre_x - image_x

            x_new = x_aux + beta * y_aux
            y_new = beta * x_aux + y_aux

            x_new = int(sheared_centre_x - x_new)
            y_new = int(sheared_centre_y - y_new)

            if 0 < x_new < width and 0 < y_new < height:
                sheared_image_x[y_new][x_new][:] = image[image_y][image_x][:]

    # resized_sheared_image_x = sheared_image_x[
    #     int(sheared_centre_y) - new_image_height // 2 : int(sheared_centre_y) + new_image_height // 2,
    #     int(sheared_centre_x) - new_image_width // 2 : int(sheared_centre_x) + new_image_width // 2,:]
    # print(resized_sheared_image_x.shape)

    if show:
        final_image = np.hstack((image, sheared_image_x))
        cv2.imshow('Images: before and after', final_image)
        cv2.waitKey()
    return sheared_image_x


if __name__ == '__main__':
    image = cv2.imread('Images/img_12.jpg')
    # flip(image, ["1","1"], show=True)
    scale(image, p=["0.5", "-2"], show=True)
    # shear(image, p=["0.5"], show=True)
