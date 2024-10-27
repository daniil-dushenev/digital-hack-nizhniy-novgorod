import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import WaterDataset, get_data_list  # Replace with your dataset class
from unet2d import UNet  # Replace with your model class
from sklearn.model_selection import train_test_split
import os
from validate import validate
from utils import save_predictions
import segmentation_models_pytorch as smp
import re

def train_test_split_files(img_dir, mask_dir, test_size=0.2, random_state=42):
    """
    Разделяет файлы изображений и масок на тренировочный и тестовый наборы на основе имен файлов.
    
    Parameters:
        img_dir (str): Путь к папке с изображениями.
        mask_dir (str): Путь к папке с масками.
        test_size (float): Доля данных, выделяемая на тестовый набор.
        random_state (int): Значение для случайного выбора файлов, чтобы результаты были воспроизводимыми.
    
    Returns:
        train_files (list): Список имен файлов для тренировочного набора.
        test_files (list): Список имен файлов для тестового набора.
    """
    # Получаем список имен файлов изображений
    img_filenames = sorted(os.listdir(img_dir))
    mask_filenames = sorted(os.listdir(mask_dir))
    
    # Убедимся, что количество изображений и масок совпадает
    assert len(img_filenames) == len(mask_filenames), "Количество изображений и масок должно совпадать"

    # Убираем расширение для удобства сопоставления файлов изображений и масок
    img_names = [os.path.splitext(f)[0] for f in img_filenames]
    mask_names = [os.path.splitext(f)[0] for f in mask_filenames]
    print(img_names)
    
    # Убедимся, что имена изображений и масок совпадают
    assert img_names == mask_names, "Имена изображений и масок должны совпадать"

    # Разделение имен файлов
    train_filenames = list(filter(lambda x: "5.tif" not in x, img_filenames))
    test_filenames = list(filter(lambda x: "5.tif" in x, img_filenames))
    # train_filenames, test_filenames = train_test_split(img_filenames, test_size=test_size, random_state=random_state)
    print(test_filenames[:10], train_filenames[::100])
    # print(train_filenames)

    return train_filenames, test_filenames

def train_model(model, train_dataloader, test_dataloader, criterion1, optimizer, num_epochs, scheduler, criterion2=None, device='cuda'):
    """
    Train a machine learning model using the provided training and validation data loaders.

    This function iterates through the specified number of epochs, performing forward and backward passes,
    updating the model parameters, and validating the model on the test dataset. It logs training and validation
    metrics and saves the model weights if the validation loss improves.

    Parameters:
    model (torch.nn.Module): The model to be trained.
    train_dataloader (DataLoader): DataLoader for the training dataset.
    test_dataloader (DataLoader): DataLoader for the validation dataset.
    criterion1 (torch.nn.Module): The loss function used for training.
    optimizer (torch.optim.Optimizer): The optimizer used for updating model parameters.
    num_epochs (int): The number of epochs to train the model.
    scheduler (torch.optim.lr_scheduler): Learning rate scheduler to adjust the learning rate.
    criterion2 (torch.nn.Module, optional): An optional second loss function for multi-task learning. Defaults to None.
    device (str, optional): The device to perform training on ('cuda' or 'cpu'). Defaults to 'cuda'.
    """
    best_f1 = 0.0
    accumulation_steps = 2
    best_loss = 10000
    log_data = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for i, (inputs, labels) in enumerate(train_dataloader):
            # Move data to the device (GPU or CPU)
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            # optimizer.zero_grad()

            # Forward pass
            # print(inputs.shape)
            outputs = model(inputs)
            # print(outputs.shape, labels.shape)
            # return False
            # preds = torch.argmax(outputs, dim=1)
            # outputs = torch.argmax(outputs, dim=1)
            # print(labels)
            loss = criterion1(outputs, labels.long())
            # if criterion2 is not None:
            #     loss2 = criterion2(outputs[:, 1], labels.float())
            #     loss = (loss + loss2)/2
            # print(labels.shape)
            # return False
            loss.backward()
            # Backward pass and optimization
            # if (i+1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()


            running_loss += loss.item()


        epoch_loss = running_loss / len(train_dataloader)
        # validate(model, test_dataloader, criterion)
        # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        val_loss, precision, recall, f1 = validate(model, test_dataloader, criterion1, device=device)

        # Log metrics
        scheduler.step()
        log_entry = f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}'
        print(log_entry)
        log_data.append(log_entry)

        # Save model weights if the F1 score is the best so far
        # if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), f'weights_unet/newC_unetplusplus_10C_resnet50_{val_loss}_{f1}.pth')
        # print(f'Saved model with best Val loss: {best_loss:.4f}')
            # save_predictions(model, test_dataloader, 'pipeline/outputs/', device)

    # Save log data to a file
    log_file = 'logs_for.txt'
    with open(log_file, 'w') as f:
        for entry in log_data:
            f.write(entry + '\n')
    print('Training complete.')
    # save_predictions(model, test_dataloader, 'pipeline/outputs/', device)


def main():
    """
    Main function to set up and execute the training of the model.

    This function initializes hyperparameters, prepares the dataset, creates data loaders, 
    initializes the model, loss function, and optimizer, and then calls the `train_model` function
    to start the training process.

    The function also manages the device setup (CPU or GPU) and handles data paths for images and masks.
    """
    # Hyperparameters
    num_epochs = 15
    batch_size = 48
    learning_rate = 0.0001

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Путь к данным
    img_path = '../skoltech_train/data/images/'
    mask_path = '../skoltech_train/data/masks/'
    original_images_path = '../skoltech_train/train/images/'

    # Разделяем данные
    train_files, test_files = train_test_split_files(img_path, mask_path, test_size=0.2)

    # Создаем датасеты и загрузчики данных
    train_dataset = WaterDataset(img_path=img_path, mask_path=mask_path, file_names=train_files, original_image_path=original_images_path)
    test_dataset = WaterDataset(img_path=img_path, mask_path=mask_path, file_names=test_files)
    print(len(train_dataset), len(test_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = smp.UnetPlusPlus(encoder_name='resnet50',in_channels=13, classes=2).to(device)

    # model = UNet(10, 2).to(device)  # You need to implement this
    # criterion1 = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    criterion1 = nn.CrossEntropyLoss()  # Adjust based on your task
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)  # Adjust step_size and gamma as needed
    # Train the model
    train_model(model, train_dataloader, test_dataloader, criterion1, optimizer, num_epochs, scheduler=scheduler)

if __name__ == '__main__':
    main()