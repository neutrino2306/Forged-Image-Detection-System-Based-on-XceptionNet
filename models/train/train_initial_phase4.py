import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torchmetrics
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder

from model4 import HiFE
from torch.optim import Adam
from torch.utils.data import DataLoader, dataset
from data_preprocess_phase4 import get_train_transform, get_val_transform


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

    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(train_data_path, val_data_path, batch_size=32)

    # 实例化模型
    model = HiFE(J=1, num_classes=2)
    # 冻结Xception的预训练权重
    set_parameter_requires_grad(model.xception, True)
    model.to(device)

    criterion = CrossEntropyLoss()
    # 设置优化器（只优化未冻结的参数）
    optimizer = Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-8)

    # 实例化度量
    accuracy = torchmetrics.Accuracy().to(device)
    precision = torchmetrics.Precision(num_classes=2, average='macro').to(device)
    recall = torchmetrics.Recall(num_classes=2, average='macro').to(device)
    f1 = torchmetrics.F1(num_classes=2, average='macro').to(device)
    auc = torchmetrics.AUROC(num_classes=2).to(device)

    # 设置TensorBoard进行监控
    tensorboard_log_dir = 'D:/Dataset/my_tensorboard_logs'
    writer = SummaryWriter(tensorboard_log_dir)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 更新度量
            running_loss += loss.item() * inputs.size(0)
            accuracy.update(outputs, labels)
            precision.update(outputs, labels)
            recall.update(outputs, labels)
            f1.update(outputs, labels)
            auc.update(outputs, labels)

        epoch_loss = running_loss / len(train_loader.dataset)
        # 记录度量
        writer.add_scalar('Training Loss', epoch_loss, epoch)
        writer.add_scalar('Training Accuracy', accuracy.compute(), epoch)
        writer.add_scalar('Training Precision', precision.compute(), epoch)
        writer.add_scalar('Training Recall', recall.compute(), epoch)
        writer.add_scalar('Training F1 Score', f1.compute(), epoch)
        writer.add_scalar('Training AUC', auc.compute(), epoch)

        # 重置度量
        accuracy.reset()
        precision.reset()
        recall.reset()
        f1.reset()
        auc.reset()

        # 验证阶段
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 更新度量
                running_loss += loss.item() * inputs.size(0)
                accuracy.update(outputs, labels)
                precision.update(outputs, labels)
                recall.update(outputs, labels)
                f1.update(outputs, labels)
                auc.update(outputs, labels)

        val_epoch_loss = running_loss / len(val_loader.dataset)

        writer.add_scalar('Validation Loss', val_epoch_loss)
        writer.add_scalar('Validation Accuracy', accuracy.compute(), epoch)
        writer.add_scalar('Validation Precision', precision.compute(), epoch)
        writer.add_scalar('Validation Recall', recall.compute(), epoch)
        writer.add_scalar('Validation F1 Score', f1.compute(), epoch)
        writer.add_scalar('Validation AUC', auc.compute(), epoch)

        # 保存模型
        torch.save(model.state_dict(), f'{save_model_path}/epoch_{epoch + 1:02d}_val_loss_{val_epoch_loss:.2f}.pt')

        accuracy.reset()
        precision.reset()
        recall.reset()
        f1.reset()
        auc.reset()

    writer.close()


if __name__ == '__main__':
    train_model('D:/Dataset/FF++_dataset_by_ls/jpg/c23/total/train', 'D:/Dataset/FF++_dataset_by_ls/jpg/c23/total/val',
                'D:/Dataset/FF++_dataset_by_ls/jpg/c23/total/')
