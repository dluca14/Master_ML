import cv2
import os
from tkinter import Tk  # from tkinter import Tk for Python 3.x
from tkinter import filedialog
from numpy.lib.function_base import add_newdoc_ufunc
from tqdm import tqdm
from augmentations import *
from geometric import *

pixel_augmentations = {
    "brightness": brightness,
    "hue": hue,
    "saturation": saturation,
    "toneMappingInverse": toneMappingInverse,
    "toneMappingSigmoid": toneMappingSigmoid,
    "gaussianBlur": gaussianBlur,
    "sharpen": sharpen,
    "histogramEqualization": histogramEqualization,
    "translate": translate,
    "rotate": rotate,
    "flip": flip,
    "scale": scale,
    "shear": shear,

}


# ************* LAB 2 ************** #

def resize_images(images):
    print(images)
    for image in tqdm(images):
        if '.DS_Store' not in image and image.shape[1] > 2025:
            img = cv2.imread(image)
            scale_percent = 30  # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            print(resized.shape)
            img_name = image.split('/')[-1]
            print(img_name)
            cv2.imwrite('../Images/' + img_name, resized)


def read_config_file(config_path):
    file = open(config_path, 'r')

    augmentation = {}
    chained_augmentation = []
    for line in file:
        stripped_line = line.strip()
        aug_part = {}
        if '+' in stripped_line:
            chained_augmentation.append(aug_part)
            augmentations_array = stripped_line.split("+")
            for aug in augmentations_array:
                aug = aug.strip()
                augmentation_type = aug.split(" ")[0]
                parameters = aug.split(" ")[1:]
                aug_part[augmentation_type] = parameters
        else:
            augmentation_type = stripped_line.split(" ")[0]
            parameters = stripped_line.split(" ")[1:]
            augmentation[augmentation_type] = parameters

    return augmentation, chained_augmentation


def get_images_path():
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    directory = filedialog.askdirectory()  # show an "Open" dialog box and return the path to the selected folder
    print(directory)
    images = sorted(os.listdir(directory))
    for i, img in enumerate(images):
        images[i] = os.path.join(directory, img)
    print(images)
    return directory, images


def create_augmentation_folder(directory):
    opened_folder = directory.split('/')[-1]
    augmentation_folder = opened_folder + '_aug'
    augmentation_folder_path = os.path.join('/'.join(directory.split('/')[:-1]), augmentation_folder)
    os.makedirs(augmentation_folder_path, exist_ok=True)
    return augmentation_folder_path


def augment_images(images, augmentation, augmentation_folder_path):
    aug_keys = augmentation.keys()
    counter = 0
    for aug in tqdm(aug_keys):
        for img in tqdm(images):
            if '.DS_Store' not in img:
                counter += 1
                image = cv2.imread(img)
                img_name = img.split('/')[-1]
                img_name = img_name.split('.')[0]
                new_img_name = img_name + '_' + aug + '_' + str(counter) + '.jpg'
                aug_image = pixel_augmentations[aug](image, augmentation[aug] if len(augmentation[aug]) != 0 else None)
                cv2.imwrite(os.path.join(augmentation_folder_path, new_img_name), aug_image)
    return counter


def augment_chained_images(images, chained_augmentations, augmentation_folder_path, counter=0):
    for _, chain in enumerate(chained_augmentations):
        aug_keys = chain.keys()
        print(chain)
        for img in images:
            if '.DS_Store' not in img:
                counter += 1
                image = cv2.imread(img)
                img_name = img.split('/')[-1]
                img_name = img_name.split('.')[0]
                new_img_name = img_name
                for aug in aug_keys:
                    new_img_name = new_img_name + '_' + aug
                    image = pixel_augmentations[aug](image, chain[aug] if len(chain[aug]) != 0 else None)
                new_img_name = new_img_name + '_' + str(counter) + '.jpg'
                cv2.imwrite(os.path.join(augmentation_folder_path, new_img_name), image)
    return counter
