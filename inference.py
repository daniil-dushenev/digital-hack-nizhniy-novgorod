import os
import rasterio
from tqdm import tqdm
from typing import List, Optional
from rasterio.windows import Window
from torch.utils.data import DataLoader
from dataset import WaterDataset
import torch
import numpy as np
import segmentation_models_pytorch as smp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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



def get_tiles_with_overlap(image_width: int, image_height: int, 
                           tile_size: int, overlap: int) -> List[Window]:
    """
    Calculate the windows for tiles with specified overlap across the image.

    Parameters:
        image_width (int): The width of the input image in pixels.
        image_height (int): The height of the input image in pixels.
        tile_size (int): The size of each tile (assumes square tiles).
        overlap (int): The number of overlapping pixels between adjacent tiles.

    Returns:
        List[Window]: A list of rasterio Window objects representing each tile.
    """
    step_size = tile_size - overlap
    tiles = []
    for y in range(0, image_height, step_size):
        for x in range(0, image_width, step_size):
            window = Window(x, y, tile_size, tile_size)
            # Adjust window if it exceeds the image bounds
            window = window.intersection(Window(0, 0, image_width, image_height))
            # print(window)
            tiles.append(window)
    return tiles


def save_tile(src_dataset: rasterio.io.DatasetReader, window: Window, 
              output_folder: str, tile_index: int, image_id: str) -> None:
    """
    Extract and save a single tile from the source dataset.

    Parameters:
        src_dataset (rasterio.io.DatasetReader): The opened rasterio dataset (the input image).
        window (Window): The window (rasterio Window object) defining the tile.
        output_folder (str): The folder where the tiles will be saved.
        tile_index (int): Index of the tile to be used for naming the file.
        image_id (int): Image id to be used for naming the file.

    Returns:
        None
    """
    transform = src_dataset.window_transform(window)
    # print(transform)
    tile_data = src_dataset.read(window=window)
    # print(tile_data)
    
    profile = src_dataset.profile
    profile.update({
        'driver': 'GTiff',
        'height': window.height,
        'width': window.width,
        'transform': transform
    })
    
    output_filename = os.path.join(output_folder, f"tile_{image_id}_{tile_index}.tif")
    with rasterio.open(output_filename, 'w', **profile) as dst:
        dst.write(tile_data)


def split_image(image_path: str, output_folder: str, mask_path: Optional[str] = None, 
                tile_size: int = 512, overlap: int = 128, image_id: str = "0") -> None:
    """
    Split a large GeoTIFF image and its corresponding mask (if provided) into tiles with overlap 
    and save them.

    Parameters:
        image_path (str): The file path of the input TIFF image.
        mask_path (Optional[str]): The file path of the corresponding mask TIFF image. If None, only image is processed.
        output_folder (str): The folder where the tiles will be saved.
        tile_size (int, optional): The size of the tiles. Default is 512x512.
        overlap (int, optional): The number of pixels to overlap between tiles. Default is 128 pixels.
        image_id (int, optional): ID of the input image to be used for naming the file. 
            Defaults to 0.

    Returns:
        None
    """
    with rasterio.open(image_path) as src_image:
        image_width = src_image.width
        image_height = src_image.height

        # Create output directories for images and masks (if available)
        images_folder = os.path.join(output_folder, 'images')
        os.makedirs(images_folder, exist_ok=True)

        if mask_path:
            masks_folder = os.path.join(output_folder, 'masks')
            os.makedirs(masks_folder, exist_ok=True)

        # Get list of tiles with overlap
        tiles = get_tiles_with_overlap(image_width, image_height, tile_size, overlap)

        # Save image tiles (and mask tiles if provided)
        if mask_path:
            with rasterio.open(mask_path) as src_mask:
                for idx, window in tqdm(enumerate(tiles)):

                    save_tile(src_image, window, images_folder, idx, image_id)
                    save_tile(src_mask, window, masks_folder, idx, image_id)
        else:
            for idx, window in tqdm(enumerate(tiles)):
                save_tile(src_image, window, images_folder, idx, image_id)


def get_model_outputs(model, dataloader, device='cuda'):
    """
    Получает предсказания модели для всего даталоадера.

    Parameters:
        model (torch.nn.Module): модель, которая будет использоваться для прогнозирования.
        dataloader (torch.utils.data.DataLoader): даталоадер, содержащий входные данные.
        device (str): устройство для вычислений ('cuda' или 'cpu').

    Returns:
        all_outputs (torch.Tensor): тензор с предсказаниями модели для всех данных.
    """
    model.eval()  # Переводим модель в режим оценки
    all_outputs = []

    with torch.no_grad():  # Отключаем градиенты для ускорения
        for inputs, _ in tqdm(dataloader, desc="Inference Progress"):  # Второй элемент (labels) игнорируется, если он не нужен
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Если выход модели многоканальный, выбираем класс с наибольшей вероятностью
            outputs = torch.argmax(outputs, dim=1)  # Используем argmax для получения предсказаний

            all_outputs.append(outputs.cpu())  # Перемещаем на CPU для хранения в списке

    # Конкатенируем все батчи в один тензор
    all_outputs = torch.cat(all_outputs, dim=0)
    
    return all_outputs


def make_tif_from_outputs(outputs: List[np.ndarray], image_width: int, image_height: int,
                          tile_size: int, overlap: int, output_path: str, transform) -> None:
    """
    Собирает итоговую маску из вырезок (outputs) с учетом overlap и сохраняет как TIFF.

    Parameters:
        outputs (List[np.ndarray]): Список вырезок с предсказаниями модели.
        image_width (int): Ширина исходного изображения в пикселях.
        image_height (int): Высота исходного изображения в пикселях.
        tile_size (int): Размер каждой вырезки.
        overlap (int): Перекрытие между вырезками.
        output_path (str): Путь для сохранения итоговой маски в формате TIFF.
    """
    print(len(outputs))
    # Шаг с учетом перекрытия
    step_size = tile_size - overlap

    # Инициализация пустой итоговой маски и счетчика для усреднения
    final_mask = np.zeros((image_height, image_width), dtype=np.float32)
    count_map = np.zeros((image_height, image_width), dtype=np.float32)

    tile_idx = 0  # Индекс для каждой вырезки
    for y in range(0, image_height, step_size):
        for x in range(0, image_width, step_size):
            # Получаем текущую вырезку из outputs
            tile = outputs[tile_idx].cpu().numpy().astype(np.float32)
            tile_idx += 1

            # Ограничиваем размеры, чтобы не выйти за пределы изображения
            tile_height, tile_width = tile.shape
            end_y = min(y + tile_height, image_height)
            end_x = min(x + tile_width, image_width)

            # Обновляем итоговую маску и счетчик для усреднения
            final_mask[y:end_y, x:end_x] += tile[:end_y-y, :end_x-x]
            count_map[y:end_y, x:end_x] += 1

    print(tile_idx)

    # Усредняем значения в местах перекрытия
    final_mask = final_mask / np.maximum(count_map, 1)  # предотвращаем деление на ноль

    # Создаем родительские директории для output_path, если они отсутствуют
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Сохраняем как TIFF с использованием профиля данных
    # with rasterio.open(
    #     output_path,
    #     'w',
    #     driver='GTiff',
    #     height=image_height,
    #     width=image_width,
    #     count=1,
    #     dtype=final_mask.dtype
    # ) as dst:
    #     dst.write(final_mask, 1)
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=image_height,
        width=image_width,
        count=1,
        dtype=np.int32,
        # crs='EPSG:4326',  # Set the coordinate reference system (modify if needed)
        transform=transform  # Modify if you have specific georeferencing
    ) as dst:
        dst.write(final_mask, 1)


def get_wh_from_image(image_path):
    """
    Get the width and height of an image.

    Parameters:
    image_path (str): The file path of the image.

    Returns:
    tuple: A tuple containing the width and height of the image (width, height).
    """
    with rasterio.open(image_path) as src_image:
        image_width = src_image.width
        image_height = src_image.height
    return image_width, image_height


def inference(image_path, output_folder, inference_path, image_id, model, tile_size=256, overlap=32, batch_size=16, device="cuda"):
    """
    Perform inference on an image using a specified model.

    This function splits the image into tiles, processes them through the model,
    and then reconstructs the output into a single TIFF file.

    Parameters:
    image_path (str): The file path of the input image.
    output_folder (str): The folder where the tiles will be saved.
    inference_path (str): The file path where the output mask will be saved.
    image_id (str): The identifier for the image (used for naming).
    model (torch.nn.Module): The model used for inference.
    tile_size (int, optional): The size of the tiles to split the image into. Defaults to 256.
    overlap (int, optional): The amount of overlap between tiles. Defaults to 32.
    batch_size (int, optional): The number of tiles to process in a batch. Defaults to 16.
    device (str, optional): The device to perform inference on (e.g., "cuda" or "cpu"). Defaults to "cuda".
    """

    transform = get_transform_from_image(image_path)
    split_image(
        image_path=image_path, mask_path=None,
        output_folder=output_folder, tile_size=256,
        overlap=32, image_id=image_id
        ) 
    output_folder += 'images/'
    img_filenames = sorted((os.listdir(output_folder)), key=natural_sort_key)

    print(img_filenames[:10])
    # print(img_filenames[:10])
    dataset = WaterDataset(img_path=output_folder, mask_path=None, file_names=img_filenames, test=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    outputs = get_model_outputs(model, dataloader, device)
    image_width, image_height = get_wh_from_image(image_path)
    make_tif_from_outputs(outputs, image_width, image_height,
                          tile_size, overlap, inference_path, transform)
    

def get_transform_from_image(image_path):
    """
    Get the transformation matrix from an image.

    Parameters:
    image_path (str): The file path of the image.

    Returns:
    affine.Affine: The transformation matrix associated with the image.
    """
    with rasterio.open(image_path) as src:
        transform = src.transform  # Извлекаем transform
    return transform


def main():
    """
    The main function that orchestrates the inference process.

    This function initializes the model, iterates through all TIFF images in the specified folder,
    and performs inference on each image, saving the results to the specified output folder.
    """
    images_folder = "test_scoltech/images/" #  folder with images
    output_folder = "data/test/" #  folder to save tiles
    inference_folder = "test_scoltech/outputs/unetplusplus_10channels_newchannel/images/" # folder to save output mask
    model_weights_path = "weights_unet/newC_unetplusplus_10C_resnet50_0.021187722799368203_0.8234627842903137.pth"
    device = "cuda"
    
    # Initialize the model
    model = smp.UnetPlusPlus(encoder_name='resnet50', in_channels=13, classes=2).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    
    # Loop through each file in the images folder
    for filename in os.listdir(images_folder):
        if filename.endswith(".tif"):  # Only process .tif files
            image_path = os.path.join(images_folder, filename)
            image_id = os.path.splitext(filename)[0]  # e.g., "9_2" from "9_2.tif"
            
            # Set the paths for output folder and inference result
            image_output_folder = os.path.join(output_folder, f"{image_id}_image/")
            inference_path = os.path.join(inference_folder, f"{image_id}.tif")
            
            # Ensure output folders exist
            os.makedirs(image_output_folder, exist_ok=True)
            
            # Perform inference on the current image
            inference(
                image_path=image_path,
                output_folder=image_output_folder,
                inference_path=inference_path,
                image_id=image_id,
                model=model
            )

if  __name__ == "__main__":
    main()
