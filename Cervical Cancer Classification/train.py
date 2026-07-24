BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20
NUM_CLASSES = 5

import torch 
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.vgg16 import VGG16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(
    root="data/SIPaKMeD/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root="data/SIPaKMeD/val",
    transform=val_transform
)

test_dataset = datasets.ImageFolder(
    root="data/SIPaKMeD/test",
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size = BATCH_SIZE,
    shuffle = True
)

val_loader = DataLoader(
    val_dataset,
    batch_size = BATCH_SIZE,
    shuffle = False
)

test_loader = DataLoader(
    test_dataset,
    batch_size = BATCH_SIZE,
    shuffle = False
)

images, labels = next(iter(train_loader))

print(images.shape)
print(labels.shape)


