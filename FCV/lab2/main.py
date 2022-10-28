from utils import *

if __name__ == '__main__':
    directory, images = get_images_path()
    augmentation, chained_augementations = read_config_file('./config_pixel.txt')
    # augmentation, chained_augementations = read_config_file('./config_geometric.txt')
    # augmentation, chained_augementations = read_config_file('./config_combined.txt')
    augmentation_folder_path = create_augmentation_folder(directory)
    counter = augment_images(images, augmentation, augmentation_folder_path)
    print(counter)
    counter = augment_chained_images(images, chained_augementations, augmentation_folder_path, counter=counter)
    print(counter)
