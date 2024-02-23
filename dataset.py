import os
from torchvision import datasets, transforms
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import KFold
from torch import nn
from torch import optim
from torch.utils.data import SubsetRandomSampler
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

split_dataset_path = "splitted/"

def get_dataset():
    train_dir = os.path.join(split_dataset_path, "train")
    val_dir = os.path.join(split_dataset_path, "val")

    train_dataset = datasets.ImageFolder(train_dir, transform=transform(),)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform())

    # Define transformations if needed
    return train_dataset, val_dataset


def transform():
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform

def make_dataloader():

    train_dataset, val_dataset = get_dataset()
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=1)
    
    return train_dataloader, val_dataloader