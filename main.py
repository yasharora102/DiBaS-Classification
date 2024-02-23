import torch
from torch import nn

from argparse import ArgumentParser
from dataset import make_dataloader

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm 
import os


def get_args():
    parser = ArgumentParser(description="Train a model on a dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=str, default=1)
    # parser.add_argument("")
    return parser.parse_args()



def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer):
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
        return model, optimizer, start_epoch, best_acc
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")
        return model, optimizer, 0, 0

def train_one_epoch(train_loader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0

    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct_predictions.double() / len(train_loader.dataset)

    print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

def validate(val_loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = correct_predictions.double() / len(val_loader.dataset)

    print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    return epoch_acc

def main():
    args = get_args()
    train_loader , val_loader = make_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(pretrained=False)
    num_classes = len(train_loader.dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 10

    checkpoint_dir = 'Checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'model_best.pth.tar')

    model, optimizer, start_epoch, best_acc = load_checkpoint(checkpoint_path, model, optimizer)

    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        train_one_epoch(train_loader, model, criterion, optimizer, device)
        val_acc = validate(val_loader, model, criterion, device)

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, filename=checkpoint_path)

    print('Training complete')

if __name__ == "__main__":
    main()