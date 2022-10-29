import json
import os

import cv2
from tkinter import Tk, filedialog

from aug_algorithms import *


def get_directory_from_user():
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    directory = filedialog.askdirectory()  # show an "Open" dialog box and return the path to the selected folder

    return directory


def apply_predefined_augmentation_algorithms(input_dir, config_data):
    k = 0
    for img_name in os.listdir(input_dir):
        k += 1
        img_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), input_dir + '/' + img_name)
        img = cv2.imread(img_path)

        for line in config_data:
            for alg_name, alg_param in line.items():
                import aug_algorithms
                func_to_run = getattr(aug_algorithms, alg_name)
                img = func_to_run(img, param=alg_param, show=False)

            new_img_name = img_name.split('.')[0] + '_aug_' + str(k) + '.jpg'
            cv2.imwrite(os.path.join(input_dir, new_img_name), img)


def save_images_to_output_folder():
    os.mkdir(input_dir + '_aug')


if __name__ == '__main__':
    f = open('config.json')
    config_data = json.load(f)['algorithms']

    input_dir = get_directory_from_user()
    # input_dir = '/home/david/workspace/Master/FCV/Proiect1_FCV/images'
    apply_predefined_augmentation_algorithms(input_dir, config_data)

    save_images_to_output_folder()
