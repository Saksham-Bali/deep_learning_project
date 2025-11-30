import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Only used for confusion matrix

from pcseg.dataset_toronto3d import Toronto3DDataset, NUM_CLASSES, CLASS_NAMES
from pcseg.model_pc_transformer import PointTransformerSeg
from pcseg.class_weights import compute_class_weights

# -------------------------
# FOCAL LOSS + CLASS BALANCE
# -------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


# -------------------------
# IOU + CONFUSION MATRIX
# -------------------------
def compute_iou(preds, labels, num_classes=NUM_CLASSES):
    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        label_cls = (labels == cls)
        inter = (pred_cls & label_cls).sum().item()
        union = (pred_cls | label_cls).sum().item()
        iou = inter / (union + 1e-6)
        ious.append(iou)
    return ious, np.mean(ious)


# -------------------------
# TRAIN (LIGHT PRINTING)
# -------------------------
def train_epoch(loader, model, criterion, optimizer, device, epoch, warmup_epochs, base_lr):
    model.train()
    total_loss = total_correct = total_points = 0

    for i, (xyz, feats, labels) in enumerate(loader):
        xyz, feats, labels = xyz.to(device), feats.to(device), labels.to(device)

        if epoch < warmup_epochs:
            lr = base_lr * float(i + 1) / (warmup_epochs * len(loader))
            for g in optimizer.param_groups:
                g["lr"] = lr

        optimizer.zero_grad()
        outputs = model(xyz, feats).permute(0, 2, 1)
        loss = criterion(outputs, labels)

        if torch.isnan(loss):  # Skip if NaN
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent gradient explosion
        optimizer.step()

        total_loss += loss.item() * xyz.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_points += labels.numel()

        if i % 50 == 0:  # only print every 50 batches
            print(f"  Batch {i}/{len(loader)} | Loss={loss.item():.4f}")

    return total_loss / total_points, total_correct / total_points


# -------------------------
# VALIDATION + CONF MATRIX
# -------------------------
def validate(loader, model, criterion, device):
    model.eval()
    total_iou = []
    total_loss = total_acc = total_points = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xyz, feats, labels in loader:
            xyz, feats, labels = xyz.to(device), feats.to(device), labels.to(device)
            outputs = model(xyz, feats).permute(0, 2, 1)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * xyz.size(0)

            preds = outputs.argmax(1)
            correct = (preds == labels).sum().item()
            total_acc += correct
            total_points += labels.numel()

            ious, mean_iou = compute_iou(preds.cpu(), labels.cpu())
            total_iou.append(mean_iou)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return (
        total_loss / total_points,
        total_acc / total_points,
        np.mean(total_iou),
        np.concatenate(all_preds),
        np.concatenate(all_labels)
    )


# -------------------------
#  VISUALIZATION AFTER TRAIN
# -------------------------
def visualize_results(preds, labels):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="magma",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Bar plot of IoU
    ious, _ = compute_iou(torch.tensor(preds), torch.tensor(labels))
    plt.figure(figsize=(8,4))
    plt.bar(CLASS_NAMES, ious)
    plt.title("Per-Class IoU")
    plt.xticks(rotation=45)
    plt.show()


# -------------------------
# MAIN TRAINING SCRIPT
# -------------------------
def main():
    os.makedirs("checkpoints_balanced", exist_ok=True)

    EPOCHS = 20
    BATCH_SIZE = 8
    LR = 5e-4
    WARMUP_EPOCHS = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = Toronto3DDataset(split="train", augment=True)
    val_ds = Toronto3DDataset(split="val", augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    weights = compute_class_weights("train").to(device)
    model = PointTransformerSeg(in_channels=6, num_classes=NUM_CLASSES).to(device)

    criterion = FocalLoss(weight=weights, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_miou = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(train_loader, model, criterion,
                                            optimizer, device, epoch, WARMUP_EPOCHS, LR)
        val_loss, val_acc, val_miou, preds, labels = validate(val_loader, model, criterion, device)

        print(f"\nEpoch {epoch}/{EPOCHS} | Train L={train_loss:.4f} | Acc={train_acc:.4f}")
        print(f"Val   L={val_loss:.4f} | Acc={val_acc:.4f} | mIoU={val_miou:.4f}")

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), "checkpoints_balanced/best_model.pth")
            print(f"*** NEW BEST MODEL SAVED (mIoU={best_miou:.4f}) ***")

    print("\nTraining Complete. Best mIoU =", best_miou)

    visualize_results(preds, labels)  # 🔥 After training


if __name__ == "__main__":
    main()
