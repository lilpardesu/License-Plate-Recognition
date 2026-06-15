import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
import numpy as np

# ==================== CONFIGURATION ====================
class Config:
    IMG_HEIGHT = 32
    IMG_WIDTH = 128
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 0.001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths - now pointing to grayscale folder
    GRAY_DIR = 'plates_gray'  # <-- Changed to grayscale folder
    SAVE_DIR = 'models/crnn'
    os.makedirs(SAVE_DIR, exist_ok=True)

# Persian chars + digits
CHARS = ['-'] + [str(i) for i in range(10)] + ['ب', 'ج', 'د', 'س', 'ص', 'ط', 'ق', 'ل', 'م', 'ن', 'و', 'ه', 'ی', 'ا', 'ت', 'پ', 'ث', 'چ', 'ح', 'خ', 'ذ', 'ر', 'ز', 'ژ', 'ش', 'ض', 'ظ', 'ع', 'غ', 'ف', 'ک', 'گ']
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARS)}
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}
NUM_CLASSES = len(CHARS)

# ==================== DATASET (GRAYSCALE) ====================
class PlateDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        
        # Filter valid
        valid = []
        for _, row in self.df.iterrows():
            if os.path.exists(os.path.join(img_dir, row['filename'])):
                valid.append(row)
        self.df = pd.DataFrame(valid)
        print(f"📁 Loaded {len(self.df)} samples from {csv_file}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        
        # Read as GRAYSCALE directly
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((Config.IMG_HEIGHT, Config.IMG_WIDTH), dtype=np.uint8)
        
        # Resize
        img = cv2.resize(img, (Config.IMG_WIDTH, Config.IMG_HEIGHT))
        
        # Normalize and add channel dim (1, H, W)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # Add channel dimension
        
        # Text to indices
        text = str(row['text'])
        indices = [CHAR_TO_IDX.get(c, 0) for c in text]
        indices = torch.tensor(indices, dtype=torch.long)
        
        return img, indices, len(indices)

def collate_fn(batch):
    images, texts, lengths = zip(*batch)
    images = torch.stack(images, 0)  # B x 1 x H x W
    texts = torch.cat(texts)
    lengths = torch.tensor(lengths)
    return images, texts, lengths

# ==================== CRNN MODEL ====================
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        
        # CNN for grayscale (1 channel input)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),  # 1 channel for grayscale
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x64
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8x32
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 1), padding=(0, 1)),  # 4x33
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 1), padding=(0, 1)),  # 2x34
            
            nn.Conv2d(512, 512, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        # RNN
        self.rnn = nn.LSTM(512, 256, 2, batch_first=True, 
                          bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # x: B x 1 x 32 x 128
        conv = self.cnn(x)  # B x 512 x 1 x 33 (roughly)
        
        # Reshape for RNN
        b, c, h, w = conv.size()
        conv = conv.squeeze(2)  # Remove height dim: B x 512 x W
        conv = conv.permute(0, 2, 1)  # B x W x 512
        
        rnn_out, _ = self.rnn(conv)
        output = self.fc(rnn_out)
        return nn.functional.log_softmax(output, dim=2)

def decode(outputs):
    """CTC decode"""
    outputs = outputs.permute(1, 0, 2)  # T x B x C
    _, max_idx = torch.max(outputs, 2)
    
    preds = []
    for b in range(max_idx.size(1)):
        seq = max_idx[:, b].cpu().numpy()
        decoded = []
        prev = -1
        for idx in seq:
            if idx != prev and idx != 0:
                decoded.append(IDX_TO_CHAR.get(idx, ''))
            prev = idx
        preds.append(''.join(decoded))
    return preds

# ==================== TRAINING ====================
def train():
    # Load grayscale data
    train_ds = PlateDataset('data/ocr/train_labels.csv', 
                           f'data/ocr/train/{Config.GRAY_DIR}')
    val_ds = PlateDataset('data/ocr/validation_labels.csv', 
                         f'data/ocr/validation/{Config.GRAY_DIR}')
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, 
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, 
                           collate_fn=collate_fn)
    
    model = CRNN(NUM_CLASSES).to(Config.DEVICE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    
    print(f"\n🚀 Training on {Config.DEVICE}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    best_loss = float('inf')
    
    for epoch in range(Config.EPOCHS):
        # Train
        model.train()
        total_loss = 0
        
        for i, (imgs, texts, lens) in enumerate(train_loader):
            imgs = imgs.to(Config.DEVICE)
            
            outputs = model(imgs)  # B x T x C
            outputs = outputs.permute(1, 0, 2)  # T x B x C
            
            T, B, _ = outputs.size()
            input_lens = torch.full((B,), T, dtype=torch.long)
            
            optimizer.zero_grad()
            loss = criterion(outputs, texts, input_lens, lens)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 20 == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"\n📊 Epoch {epoch+1}/{Config.EPOCHS} | Avg Loss: {avg_loss:.4f}")
        
        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model': model.state_dict(),
                'chars': CHAR_TO_IDX,
                'config': {
                    'IMG_HEIGHT': Config.IMG_HEIGHT,
                    'IMG_WIDTH': Config.IMG_WIDTH,
                    'BATCH_SIZE': Config.BATCH_SIZE,
                    'EPOCHS': Config.EPOCHS,
                    'LR': Config.LR,
                    'GRAY_DIR': Config.GRAY_DIR
                }
            }, f'{Config.SAVE_DIR}/crnn_best.pt')

            print("💾 Saved best model")
    
    print("\n✅ Training complete!")

if __name__ == "__main__":
    train()
