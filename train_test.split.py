import os
import shutil
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Define the path to your dataset
dataset_path = "Original/"

# Define the path where the split dataset will be stored
# It's important not to mix the original dataset, so we create a new directory
split_dataset_path = "splitted/"
os.makedirs(split_dataset_path, exist_ok=True)

train_dir = os.path.join(split_dataset_path, "train")
val_dir = os.path.join(split_dataset_path, "val")

# Split ratio for your validation set
val_split_ratio = 0.2

# Define transformations if needed
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def split_dataset(dataset_path, train_dir, val_dir, val_split_ratio):
    classes = os.listdir(dataset_path)
    for cls in tqdm(classes):
        print(f"Splitting class: {cls}")
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

        # List all files in the current class directory
        files = os.listdir(os.path.join(dataset_path, cls))
        train_files, val_files = train_test_split(files, test_size=val_split_ratio)

        # Copy training files
        for file in train_files:
            shutil.copy(
                os.path.join(dataset_path, cls, file),
                os.path.join(train_dir, cls, file),
            )

        # Copy validation files
        for file in val_files:
            shutil.copy(
                os.path.join(dataset_path, cls, file), os.path.join(val_dir, cls, file)
            )


# Split the dataset
split_dataset(dataset_path, train_dir, val_dir, val_split_ratio)

# Load the datasets and create DataLoaders
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

batch_size = 32
num_workers = 4

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)

# Now, train_loader and val_loader are ready to be used in your training loop.
