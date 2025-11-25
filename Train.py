import torch
import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

from models import MFNet
from Dataset_and_utils import MultiSpectralDataset, combined_loss, iou_score, dice_loss


# Device configuration
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "/content/mfnet(big)_checkpoint.pth"
CHECKPOINT_PATH_drive = "/content/drive/MyDrive/mfnet(big)_checkpoint.pth"
BEST_MODEL_PATH = "/content/best_model.pth"
BEST_DICE_MODEL_PATH = "/content/best_dice_model.pth"

####################
# Training Setup
####################
model = MFNet(in_ch=16, n_class=1, use_se=True, deep_supervision=True).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)   # changing lr from 3e-4 to 1e-4
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.65, patience=3, min_lr=1e-11) #changing min_lr from 1e-6 to 5e-7

if os.path.exists(CHECKPOINT_PATH):
    try:
        checkpoint_1 = torch.load(BEST_MODEL_PATH, map_location=device)
        checkpoint_2 = torch.load(BEST_DICE_MODEL_PATH, map_location=device)

        model.load_state_dict(checkpoint_1['model_state_dict'])
        optimizer.load_state_dict(checkpoint_2['optimizer_state_dict'])
        start_epoch = checkpoint_1['epoch'] +1
        print(f"Loaded checkpoint from epoch {checkpoint_1['epoch']}")
    except Exception as e:
        print(f"Failed to load checkpoint \n({e}). \nStarting from scratch.")
else:
    print("No checkpoint found - starting anew.")

#################
# Data Loader
#################
DATA_DIR = "/root/.cache/kagglehub/competitions/kaggle-competition-dl-f-2025"
X_train = np.load('/root/.cache/kagglehub/competitions/kaggle-competition-dl-f-2025/X_train_256.npy', mmap_mode='r')
Y_train = np.load('/root/.cache/kagglehub/competitions/kaggle-competition-dl-f-2025/Y_train_256.npy', mmap_mode='r')

print("Train shape:", X_train.shape)
print("Train mask shape:", Y_train.shape, "\n")

indices = np.arange(len(X_train))
train_idx, val_idx = train_test_split(indices, test_size=0.25, random_state=303)
del indices
train_ds = MultiSpectralDataset(X_train, Y_train, train_idx, augment=True, compute_stats=True)
val_ds   = MultiSpectralDataset(X_train, Y_train, val_idx, augment=False, compute_stats=True)
del X_train, Y_train
BATCH_SIZE = 24
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

def train_model(start_epoch=0,EPOCHS=300):

    best_val_iou = 0.0
    best_val_dice = 0.0

    for epoch in range(start_epoch, EPOCHS):

        # ------------------------ TRAIN ------------------------
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{EPOCHS}")

        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)

            loss = combined_loss(logits, masks)
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        avg_train_loss = total_loss / len(train_loader)

        # ------------------------ VALIDATION ------------------------
        model.eval()
        val_iou_total = 0.0
        val_count = 0
        total_dice = 0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.float()
                masks = masks.to(device)

                logits, _, _ = model(imgs)

                # IoU @ threshold=0.1 (probabilities -> binary mask inside function)
                iou = iou_score(logits, masks)
                dice = dice_loss(logits, masks)            # both on GPU

                total_dice += (1-dice)

                if not math.isnan(iou):
                    val_iou_total += iou
                    val_count += 1

        val_iou = (val_iou_total / val_count) if val_count > 0 else 0.0
        val_dice = (total_dice / val_count) if val_count > 0 else 0.0

        # ------------------------ SCHEDULER (on Dice) ------------------------
        scheduler.step((val_dice))

        # ------------------------ PRINT STATUS ------------------------
        print(
            f"Epoch {epoch+1}: "
            f"TrainLoss={avg_train_loss:.4f}, "
            f"Val IoU={val_iou:.4f}, "
            f"Val Dice ={val_dice:.4f}, "
            f"LR={optimizer.param_groups[0]['lr']:.2e}"
        )

        # ------------------------ SAVE CHECKPOINT EVERY EPOCH ------------------------
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, CHECKPOINT_PATH)

        # ------------------------ BEST IoU SAVE ------------------------
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(checkpoint, BEST_MODEL_PATH)
            print(f"New BEST IoU model saved (Val IoU={best_val_iou:.4f})\n")

        # ------------------------ BEST DICE SAVE ------------------------
        if (val_dice) > best_val_dice:
            best_val_dice = val_dice
            torch.save(checkpoint, BEST_DICE_MODEL_PATH)
            print(f"New BEST Dice model saved (Val Dice={best_val_dice:.4f})\n")

    print(f"Best Validation IoU: {best_val_iou:.4f}")
    print(f"Best Validation Dice: {best_val_dice:.4f}")

train_model(start_epoch=0, EPOCHS=150)  #for better training, train till 300 is suggested