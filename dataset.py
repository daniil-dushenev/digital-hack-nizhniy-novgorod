import os
import rasterio
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

# from PIL import Image
from torch.utils.data import Dataset
# from torch.utils.data import DataLoader

import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from PIL import Image
import re

def natural_sort_key(s):
    """
    Generate a sort key for a string that contains numbers.
    
    Parameters:
        s (str): The string to generate a sort key for.
    
    Returns:
        list: A list of strings and integers for sorting.
    """
    # Split the string into a list of strings and integers
    return [int(text) if text.isdigit() else text for text in re.split('(\d+)', s)]


def process_and_save_image(image_path, output_folder):
    """
    Reads a TIFF image, normalizes its RGB channels, and saves the processed image.

    Parameters:
    - image_path: str, path to the input TIFF image.
    - output_folder: str, path to the folder where the processed image will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Read the image using Rasterio
    with rasterio.open(image_path) as src:
        # Read the three channels (assuming RGB)
        red = src.read(1)
        green = src.read(2)
        blue = src.read(3)

    # Stack the channels into a single array
    rgb_image = np.stack((red, green, blue), axis=-1)

    # Normalize the image to the range [0, 255]
    rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image)) * 255
    rgb_image = rgb_image.astype(np.uint8)
    # print(rgb_image)
    # rgb_image = rgb_image.resize((256, 256))
    # Create a filename for the output image
    base_name = os.path.basename(image_path)
    file_name = os.path.splitext(base_name)[0] + '_normalized.png'
    output_path = os.path.join(output_folder, file_name)
    # print(rgb_image)

    # Save the normalized image using Pillow
    Image.fromarray(rgb_image).resize((256, 256)).save(output_path)

    print(f"Processed image saved at: {output_path}")

# Example usage:
    

def image_padding(image, target_size=256):
    """
    Pad an image to a target size using reflection padding.
    """
    height, width = image.shape[1:3]
    pad_height = max(0, target_size - height)
    pad_width = max(0, target_size - width)
    padded_image = np.pad(image, ((0, 0), (0, pad_height),
                                  (0, pad_width)), mode='reflect')
    return padded_image


def mask_padding(mask, target_size=256):
    """
    Pad a mask to a target size using reflection padding.
    """
    height, width = mask.shape
    pad_height = max(0, target_size - height)
    pad_width = max(0, target_size - width)
    padded_mask = np.pad(mask, ((0, pad_height), (0, pad_width)),
                         mode='reflect')
    return padded_mask

def get_data_list(img_path):
    """
    Retrieves a list of file names from the given directory.
    """
    name = []
    for _, _, filenames in os.walk(img_path): # given a directory iterates over the files
        for filename in filenames:
            f = filename.split('.')[0]
            name.append(f)

    df =  pd.DataFrame({'id': name}, index = np.arange(0, len(name))
                       ).sort_values('id').reset_index(drop=True)
    df = df['id'].values

    return np.delete(df, 0)

class WaterDataset(Dataset):
    def __init__(self, img_path, file_names, mask_path=None, test=False):
        self.img_path = img_path
        self.mask_path = mask_path
        self.file_names = (file_names)
        # self.original_image_path = original_image_path
        self.original_images = {}

        if test == True:
            self.numbers = ['1', '2', '3', '4']
            output_folder = 'test_scoltech/resized_images/'
            for i, name in enumerate(sorted(os.listdir(output_folder), key=natural_sort_key)):
                self.original_images[self.numbers[i]] = np.array(Image.open(output_folder+name))
                print(name, self.numbers[i])
        else:
            self.numbers = ['1', '2', '4']
            output_folder = 'outputs/original_resized_images/'
            for i, name in enumerate(sorted(os.listdir(output_folder), key=natural_sort_key)):
                self.original_images[self.numbers[i]] = np.array(Image.open(output_folder+name))
                print(name, self.numbers[i])

    def __len__(self):
            return len(self.file_names)

    def __getitem__(self, idx):
        with rasterio.open(self.img_path + self.file_names[idx]) as fin:
            image = fin.read()
        # print(image.shape)
        image = image_padding(image).astype(np.float32)
        # print(image[1]+image[6])
        ndwi_channel = (image[1]-image[6])/(image[1]+image[6])
        # print(self.file_names[idx], self.img_path)
        image[6] = ndwi_channel#np.concatenate((image, np.expand_dims(ndwi_channel, axis=0)), axis=0)
        if self.file_names[idx][5] in "1 2 3 5 4":
            orig_image_channel = self.original_images[self.file_names[idx][5]]
        else:
            orig_image_channel = self.original_images[self.file_names[idx][5:8]]
        # print(self.file_names[idx], self.img_path)
        # print(np.transpose(orig_image_channel, (2, 0, 1)).shape)
        
        image = np.concatenate((image, (np.transpose(orig_image_channel, (2, 0, 1)))), axis=0)
        # print(image.shape)

        if self.mask_path is not None:
            # Препроцессинг и аугементации
            with rasterio.open(self.mask_path + self.file_names[idx]) as fin:
                mask = fin.read(1)
            mask = mask_padding(mask)
            # print(image.dtype, mask.dtype)
            # print(mask.astype('i1').dtype)

            return image, mask.astype('i1')
        else:
            return image, 0