import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from glob import glob
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# ========== CONFIGURATION ==========
DATA_DIR = "dataset/raw_images"  # Change this path to your dataset folder
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== STEP 1: DATA PREPROCESSING ==========
def prepare_dataset(data_dir, test_size=0.2):
    """Creates a CSV file with image paths and labels."""
    image_files = glob(os.path.join(data_dir, "*.png"))
    
    labels = []
    for img_path in image_files:
        if "tumor" in img_path.lower():
            labels.append(1)
        else:
            labels.append(0)
    
    df = pd.DataFrame({"image_path": image_files, "label": labels})
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=42)
    
    train_df.to_csv("dataset/train.csv", index=False)
    test_df.to_csv("dataset/test.csv", index=False)
    print("Dataset prepared! Train:", len(train_df), " Test:", len(test_df))

# ========== STEP 2: CUSTOM DATASET CLASS ==========
class PathologyDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ========== STEP 3: DATA AUGMENTATION ==========
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ========== STEP 4: LOAD DATA ==========
def get_data_loaders():
    train_dataset = PathologyDataset("dataset/train.csv", train_transforms)
    test_dataset = PathologyDataset("dataset/test.csv", test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, test_loader

# ========== STEP 5: BUILDING RESNET MODEL ==========
class PathologyClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(PathologyClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# ========== STEP 6: TRAINING FUNCTION ==========
def train_model(model, train_loader, criterion, optimizer):
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss / len(train_loader)}")

# ========== STEP 7: EVALUATION FUNCTION ==========
def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print(classification_report(y_true, y_pred))
    print(f"AUC Score: {roc_auc_score(y_true, y_pred)}")

# ========== STEP 8: MAIN FUNCTION ==========
def main():
    print("🚀 Preparing dataset...")
    prepare_dataset(DATA_DIR)

    print("📊 Loading data...")
    train_loader, test_loader = get_data_loaders()

    print("🔍 Initializing model...")
    model = PathologyClassifier().to(DEVICE)

    # MULTI-GPU SETUP
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("🎯 Starting training...")
    train_model(model, train_loader, criterion, optimizer)

    print("📈 Evaluating model...")
    evaluate_model(model, test_loader)

    print("💾 Saving model...")
    torch.save(model.state_dict(), "pathology_model.pth")

if __name__ == "__main__":
    main()
