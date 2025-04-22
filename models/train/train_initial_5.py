import os
from datetime import datetime, timedelta

import torch
torch.autograd.set_detect_anomaly(True)
from data_preprocessing_phase5 import get_train_transform, get_val_transform
import torchmetrics
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from model5 import HiFE
from torch.optim import Adam
from torch.utils.data import DataLoader


def create_dataloaders(train_data_path, val_data_path, batch_size=32):
    # Import transformations
    train_transform = get_train_transform()
    val_transform = get_val_transform()

    # Create datasets
    train_dataset = ImageFolder(root=train_data_path, transform=train_transform)
    val_dataset = ImageFolder(root=val_data_path, transform=val_transform)

    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def train_model(train_data_path, val_data_path, save_model_path, num_epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(os.listdir(train_data_path))
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(train_data_path, val_data_path, batch_size=32)
    print(f'Total training images: {len(train_loader.dataset)}')
    print(f'Total validation images: {len(val_loader.dataset)}')

    # 实例化模型
    model = HiFE(J=1, num_classes=2)
    # 冻结Xception的预训练权重
    set_parameter_requires_grad(model.xception, True)
    model.to(device)

    criterion = CrossEntropyLoss()
    # 设置优化器（只优化未冻结的参数）
    optimizer = Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-8)

    # 实例化度量
    metrics = {
        'accuracy': torchmetrics.Accuracy(num_classes=2, average='macro', task='binary').to(device),
        'precision': torchmetrics.Precision(num_classes=2, average='macro', task='binary').to(device),
        'recall': torchmetrics.Recall(num_classes=2, average='macro', task='binary').to(device),
        'f1': torchmetrics.F1Score(num_classes=2, average='macro', task='binary').to(device),
        'auc': torchmetrics.AUROC(num_classes=2, task='binary').to(device)
    }
    # 设置TensorBoard进行监控
    tensorboard_log_dir = 'D:/Dataset/FF++_dataset_by_ls/jpg/c23/total/my_tensorboard_logs_initial_phase20'
    writer = SummaryWriter(tensorboard_log_dir)
    start_time = datetime.now()
    print('Starting: ' + start_time.strftime('%Y-%m-%d %H:%M:%S') + "\n")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} Training', leave=True)

        for batch_index, (inputs, labels) in enumerate(train_loop, start=1):
        # for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 更新总损失和样本数量
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            for name, metric in metrics.items():
                metric.update(preds, labels)

            if batch_index % 10 == 0:  # Log every 10 batches
                writer.add_scalar('Training Loss', running_loss / batch_index, epoch * len(train_loader) + batch_index)
                for name, metric in metrics.items():
                    writer.add_scalar(f'Training {name}', metric.compute(), epoch * len(train_loader) + batch_index)

            current_loss = running_loss / total_samples
            train_loop.set_postfix(loss=current_loss, acc=metrics['accuracy'].compute().item(),
                               precision=metrics['precision'].compute().item(),
                               recall=metrics['recall'].compute().item(),
                               f1=metrics['f1'].compute().item(),
                               auc=metrics['auc'].compute().item())

        elapsed_time = datetime.now() - start_time
        estimated_time = elapsed_time / (epoch + 1) * (num_epochs - epoch - 1)
        print(f'Estimated time remaining: {estimated_time}')

        # epoch_loss = running_loss / len(train_loader.dataset)
        # 记录度量
        for name, metric in metrics.items():
            writer.add_scalar(f'Training {name.capitalize()}', metric.compute(), epoch)
            metric.reset()

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        total_val_samples = 0
        val_loop = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} Validation', leave=True)

        for inputs, labels in val_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                total_val_samples += inputs.size(0)

                # 更新度量
                for name, metric in metrics.items():
                    metric.update(preds, labels)

                # Logging to TensorBoard
                if (batch_index % 10 == 0) or (batch_index == len(val_loader)):  # 确保最后一个批次也被记录
                    writer.add_scalar('Validation Loss', val_running_loss / batch_index,
                                        epoch * len(val_loader) + batch_index)
                    for name, metric in metrics.items():
                        writer.add_scalar(f'Validation {name}', metric.compute(),
                                            epoch * len(val_loader) + batch_index)

                current_val_loss = val_running_loss / total_val_samples
                val_loop.set_postfix(loss=current_val_loss, acc=metrics['accuracy'].compute().item(),
                             precision=metrics['precision'].compute().item(),
                             recall=metrics['recall'].compute().item(),
                             f1=metrics['f1'].compute().item(),
                             auc=metrics['auc'].compute().item())

        # 记录验证周期结果到TensorBoard
        for name, metric in metrics.items():
            writer.add_scalar(f'Validation {name.capitalize()}', metric.compute(), epoch)
            metric.reset()

        # 保存模型
        # torch.save(model.state_dict(), f'{save_model_path}/epoch_{epoch + 1:02d}_val_loss_{val_epoch_loss:.2f}.pt')
    torch.save(model.state_dict(),
               os.path.join(save_model_path, f'model_epoch_{epoch + 1:03d}_val_loss_{current_val_loss:.4f}.pt'))

    writer.close()


if __name__ == '__main__':
    train_model('D:/Dataset/FF++_dataset_by_ls/jpg/c23/total/train', 'D:/Dataset/FF++_dataset_by_ls/jpg/c23/total/val', 'D:/Dataset/FF++_dataset_by_ls/jpg/c23/total/saved_model_initial_phase20')
