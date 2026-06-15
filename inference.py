import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO
import numpy as np

# ---------- CRNN definition (must match training) ----------
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2,2),(2,1),padding=(0,1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2,2),(2,1),padding=(0,1)),
            nn.Conv2d(512, 512, 2), nn.BatchNorm2d(512), nn.ReLU(),
        )
        self.rnn = nn.LSTM(512, 256, 2, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        b,c,h,w = conv.size()
        conv = conv.squeeze(2).permute(0,2,1)  # B,W,512
        rnn_out,_ = self.rnn(conv)
        out = self.fc(rnn_out)
        return torch.log_softmax(out, dim=2)

def ctc_decode(logits, idx_to_char):
    # logits: B,T,C
    pred = torch.argmax(logits, dim=2)[0].cpu().numpy()  # first sample
    text, prev = [], -1
    for p in pred:
        if p != prev and p != 0:
            text.append(idx_to_char.get(int(p), ''))
        prev = p
    return ''.join(text)

def preprocess_plate(plate_bgr, img_w=128, img_h=32):
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (img_w, img_h))
    x = gray.astype(np.float32) / 255.0
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # 1,1,H,W
    return x

def load_crnn(path, device):
    ckpt = torch.load(path, map_location=device)
    char_to_idx = ckpt['chars']
    idx_to_char = {v:k for k,v in char_to_idx.items()}
    model = CRNN(num_classes=len(char_to_idx)).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, idx_to_char

def run(image_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    yolo = YOLO('ir_plate_detector/weights/best.pt')
    crnn, idx_to_char = load_crnn('models/crnn/crnn_best.pt', device)

    img = cv2.imread(image_path)
    res = yolo(img, verbose=False)[0]

    if res.boxes is None or len(res.boxes) == 0:
        print("No plate detected")
        return

    # choose highest confidence box
    confs = res.boxes.conf.cpu().numpy()
    i = int(np.argmax(confs))
    x1,y1,x2,y2 = res.boxes.xyxy[i].cpu().numpy().astype(int)

    plate = img[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
    if plate.size == 0:
        print("Invalid crop")
        return

    inp = preprocess_plate(plate).to(device)
    with torch.no_grad():
        logits = crnn(inp)  # B,T,C
    text = ctc_decode(logits, idx_to_char)

    print("Predicted plate text:", text)

if __name__ == "__main__":
    run("data/test/day_00010.jpg")  # change this
