import json
import os

import cv2
from tkinter import Tk, filedialog

from aug_algorithms import *


def get_directory_from_user():
    # directory = input("Enter username:")
    # print("Directory is: " + username)

    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    directory = filedialog.askdirectory()  # show an "Open" dialog box and return the path to the selected folder

    return directory


def apply_predefined_augmentation_algorithms(input_dir, config_data):
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), input_dir + '/' + img_name)
        img = cv2.imread(img_path)

        # rotate(img, angle=[45], show=True)
        # flip(img, p=[1, 1], show=True)
        # translate(img, p=[100, 200], show=True)
        # hue(img, p=90, show=True)
        # saturation(img, p=23, show=True)
        img_aug = brightness(img, p=90, show=True)
        flip(img_aug, p=[1, 1], show=True)


def save_images_to_output_folder():
    os.mkdir(input_dir + '_aug')


if __name__ == '__main__':
    f = open('config.json')
    config_data = json.load(f)

    input_dir = get_directory_from_user()
    # input_dir = 'images'
    apply_predefined_augmentation_algorithms(input_dir, config_data)

    save_images_to_output_folder()
