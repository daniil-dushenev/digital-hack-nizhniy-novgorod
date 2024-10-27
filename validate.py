import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score

def validate(model, dataloader, criterion,criterion2 = None, device='cuda'):
    """
    Функция для оценки модели на валидационном наборе данных.

    Параметры:
    model (torch.nn.Module): Модель, которую необходимо оценить.
    dataloader (torch.utils.data.DataLoader): Даталоадер для валидационного набора данных.
    criterion (torch.nn.Module): Функция потерь для вычисления потерь.
    criterion2 (torch.nn.Module, optional): Дополнительная функция потерь для комбинированной оценки. По умолчанию None.
    device (str): Устройство ('cuda' или 'cpu'), на котором будет выполняться оценка. По умолчанию 'cuda'.

    Возвращает:
    tuple: Средняя потеря, средняя точность, средний отзыв и средний F1-меры на валидационном наборе.
    """

    model.eval()  # Устанавливаем режим валидации
    running_loss = 0.0
    running_precision = 0.0
    running_recall = 0.0
    running_f1 = 0.0
    total_samples = 0

    with torch.no_grad():  # Отключаем градиенты для ускорения вычислений
        for inputs, labels in dataloader:
            # Перемещаем данные на выбранное устройство
            inputs, labels = inputs.to(device), labels.to(device)

            # Прогоняем данные через модель
            outputs = model(inputs.squeeze(0))

            # Рассчитываем функцию потерь
            # outputs = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, labels.long())
            # loss = criterion(outputs[:, 1], labels)
            if criterion2 is not None:
                loss += criterion2(outputs[:, 1], labels.float())
                loss /= 2
            running_loss += loss.item()

            # Переводим выходы модели в бинарные предсказания
            # print(outputs.shape)
            preds = torch.argmax(outputs, dim=1)

            # Вычисляем TP, FP, FN для метрик
            # print(labels.shape, preds.shape)
            TP = torch.sum((preds == 1) & (labels == 1)).float()
            FP = torch.sum((preds == 1) & (labels == 0)).float()
            FN = torch.sum((preds == 0) & (labels == 1)).float()
            
            # Вычисляем Precision, Recall, F1
            precision = TP / (TP + FP + 1e-8) if (TP + FP) > 0 else 0  # добавляем малое значение для избежания деления на 0
            recall = TP / (TP + FN + 1e-8) if (TP + FN) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8) if (precision + recall) > 0 else 0
            
            running_precision += precision
            running_recall += recall
            running_f1 += f1
            total_samples += 1

    # Усредняем потери и метрики на всем валидационном наборе
    avg_loss = running_loss / total_samples
    avg_precision = running_precision / total_samples
    avg_recall = running_recall / total_samples
    avg_f1 = running_f1 / total_samples

    print(f'Validation Loss: {avg_loss:.4f}, '
          f'Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}')
    return avg_loss, avg_precision, avg_recall, avg_f1
