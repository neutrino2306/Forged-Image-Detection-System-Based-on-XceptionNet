import os
from datetime import datetime, timedelta

import torch
torch.autograd.set_detect_anomaly(True)
import torchmetrics
from torch.nn import CrossEntropyLoss
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torchvision import transforms
from model5 import HiFE
from torch.optim import Adam
from torch.utils.data import DataLoader


def create_dataloaders(test_data_path, batch_size=32):
    # 定义测试时的图像转换操作
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载测试数据集
    test_dataset = ImageFolder(root=test_data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def test_model(model_path, test_data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = create_dataloaders(test_data_path, batch_size=32)
    print(f'Total test images: {len(test_loader.dataset)}')

    # 加载模型
    model = HiFE(J=1, num_classes=2)
    if model_path:
        # 加载预训练模型
        model.load_state_dict(torch.load(model_path))
        print("Pretrained model loaded.")
    model = model.to(device)

    # 定义损失函数和指标
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-8)

    # 实例化度量
    metrics = {
        'accuracy': torchmetrics.Accuracy(num_classes=2, average='macro', task='binary').to(device),
        'precision': torchmetrics.Precision(num_classes=2, average='macro', task='binary').to(device),
        'recall': torchmetrics.Recall(num_classes=2, average='macro', task='binary').to(device),
        'f1': torchmetrics.F1Score(num_classes=2, average='macro', task='binary').to(device),
        'auc': torchmetrics.AUROC(num_classes=2, task='binary').to(device),
        'confusion_matrix': torchmetrics.ConfusionMatrix(num_classes=2, task='binary').to(device)
    }

    start_time = datetime.now()
    print('Starting: ' + start_time.strftime('%Y-%m-%d %H:%M:%S') + "\n")

    model.eval()
    total_test_loss = 0.0
    total_test_samples = 0

    with torch.no_grad():
        test_loop = tqdm(test_loader, desc="Testing", leave=True)
        for inputs, labels in test_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            loss = criterion(outputs, labels)

            total_test_loss += loss.item() * inputs.size(0)
            total_test_samples += inputs.size(0)

            for name, metric in metrics.items():
                metric.update(preds, labels)

            current_test_loss = total_test_loss / total_test_samples
            test_loop.set_postfix(loss=current_test_loss, acc=metrics['accuracy'].compute().item(),
                                  precision=metrics['precision'].compute().item(),
                                  recall=metrics['recall'].compute().item(),
                                  f1=metrics['f1'].compute().item(),
                                  auc=metrics['auc'].compute().item())
    # 输出最终的测试结果
    print('Final Test Metrics:')
    for name, metric in metrics.items():
        if name == 'confusion_matrix':
            print(f'{name.capitalize()}: \n{metric.compute().cpu().numpy()}')
        else:
            value = metric.compute()
            print(f'{name.capitalize()}: {value:.4f}')

    print('Testing completed.')


if __name__ == '__main__':
    # 执行测试
    test_model('D:/Dataset/FF++_dataset_by_ls/jpg/c23/total/saved_model_further_phase21/model_epoch_005_val_loss_0.8126.pt',
                'D:/Dataset/test',
                )
