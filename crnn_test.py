import torch
import torch.nn as nn
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Persian digits + Persian letters + @ for handicap
PERSIAN_DIGITS = '۰۱۲۳۴۵۶۷۸۹'
PERSIAN_LETTERS = 'ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی'
CHARS = PERSIAN_DIGITS + PERSIAN_LETTERS + '@'
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i: c for c, i in CHAR_TO_IDX.items()}
NUM_CLASSES = len(CHARS)

def normalize_to_persian(text):
    """Convert ASCII digits to Persian digits"""
    trans = str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹')
    return text.translate(trans)

class FixedLengthCRNN(nn.Module):
    def __init__(self, num_chars=8):
        super().__init__()
        self.num_chars = num_chars
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 1),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 1),
            nn.Conv2d(512, 512, 2), nn.ReLU()
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, num_chars))
        self.classifier = nn.Linear(512, NUM_CLASSES)
        
    def forward(self, x):
        features = self.cnn(x)
        features = self.pool(features)
        features = features.squeeze(2).permute(0, 2, 1)
        return self.classifier(features)

class PlateDataset(Dataset):
    def __init__(self, csv_file='labels.csv', img_dir='data/results', augment=True):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.augment = augment
        
        # Normalize labels to Persian digits immediately
        self.data['text'] = self.data['text'].astype(str).apply(normalize_to_persian)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = f"{self.img_dir}/{row['filename']}"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return None
        
        # Safety check: ensure grayscale even if imread ignored the flag
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if self.augment:
            if np.random.random() > 0.5:
                angle = np.random.uniform(-3, 3)
                h, w = img.shape
                M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), borderValue=127)
        
        img = cv2.resize(img, (128, 32))
        img = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
        
        # Ensure exactly 8 characters, pad with Persian zero '۰' if needed
        text = row['text'][:8].ljust(8, '۰')
        labels = [CHAR_TO_IDX.get(c, 0) for c in text[:8]]
        
        return torch.FloatTensor(img).unsqueeze(0), torch.LongTensor(labels)

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None
    images, labels = zip(*batch)
    return torch.stack(images), torch.stack(labels)

# =============================================================================
# TRAINING CODE
# =============================================================================
if __name__ == "__main__":
    device = torch.device('cpu')
    model = FixedLengthCRNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    dataset = PlateDataset(augment=True)
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=4, collate_fn=collate_fn)

    print(f"Training: {len(train_set)}, Val: {val_size}")
    print(f"Characters: {CHARS[:20]}... (total: {NUM_CLASSES})")

    best_acc = 0

    for epoch in range(100):
        model.train()
        train_loss = 0
        
        for images, labels in train_loader:
            if images is None:
                continue
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs.view(-1, NUM_CLASSES), labels.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        correct_chars = 0
        total_chars = 0
        correct_plates = 0
        total_plates = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                if images is None:
                    continue
                    
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=2)
                
                correct_chars += (preds == labels).sum().item()
                total_chars += labels.numel()
                correct_plates += (preds == labels).all(dim=1).sum().item()
                total_plates += images.size(0)
                
                if epoch % 10 == 0:
                    for i in range(min(1, images.size(0))):
                        pred_str = ''.join([IDX_TO_CHAR[p.item()] for p in preds[i]])
                        true_str = ''.join([IDX_TO_CHAR[l.item()] for l in labels[i]])
                        print(f"  {true_str} -> {pred_str}")
        
        char_acc = correct_chars / total_chars * 100
        plate_acc = correct_plates / total_plates * 100
        
        print(f"Epoch {epoch}: Loss={train_loss/len(train_loader):.3f}, "
              f"Char={char_acc:.1f}%, Plate={plate_acc:.1f}%")
        
        if plate_acc > best_acc:
            best_acc = plate_acc
            torch.save(model.state_dict(), 'crnn_persian_best.pt')
            print(f"  -> Saved! Best: {best_acc:.1f}%")

    print(f"Done. Best accuracy: {best_acc:.1f}%")
