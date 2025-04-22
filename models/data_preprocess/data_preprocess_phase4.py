from torchvision import transforms


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.05, contrast=0.2, saturation=0.5, hue=0.08)
        ], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
