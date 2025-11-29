import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from dataset import Toronto3DDataset
from model import RandLANet
from loss import CombinedLoss
from utils import compute_iou

def train(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset
    train_dataset = Toronto3DDataset(
        root_dir=args.data_path, 
        split='train', 
        num_points=args.num_points, 
        block_size=args.block_size,
        sample_rate=args.sample_rate
    )
    val_dataset = Toronto3DDataset(
        root_dir=args.data_path, 
        split='val', 
        num_points=args.num_points, 
        block_size=args.block_size,
        sample_rate=args.sample_rate
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)

    # Model
    # Input dim = 6 (RGB + NormXYZ)
    # Output dim = 9 (Classes in Toronto-3D: Unclassified, Ground, Road_markings, Natural, Building, Utility_line, Pole, Car, Fence)
    model = RandLANet(d_in=6, num_classes=9, num_points=args.num_points).to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)

    # Loss
    # Calculate class weights if needed (omitted for simplicity, can be added)
    criterion = CombinedLoss().to(device)

    best_miou = 0.0
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        model.train()
        train_loss = 0.0
        train_pred = []
        train_true = []
        
        for points, features, labels in tqdm(train_loader, desc="Training"):
            points = points.to(device)
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            outputs = model(points, features) # [B, num_classes, N] (after permute inside model? No, model output is [B, N, C] or [B, C, N]?)
            # Let's check model output.
            # Model ends with fc_final: Linear(32, num_classes) -> [B, N, num_classes]
            # Loss expects [B, C, N] for CrossEntropy usually.
            
            outputs = outputs.permute(0, 2, 1) # [B, C, N]
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # Metrics
            preds = torch.argmax(outputs, dim=1) # [B, N]
            train_pred.append(preds.detach().cpu())
            train_true.append(labels.detach().cpu())
            
        scheduler.step()
        
        train_loss /= len(train_loader)
        train_pred = torch.cat(train_pred, dim=0)
        train_true = torch.cat(train_true, dim=0)
        _, train_miou = compute_iou(train_pred, train_true, 9)
        
        print(f"Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_pred = []
        val_true = []
        
        with torch.no_grad():
            for points, features, labels in tqdm(val_loader, desc="Validation"):
                points = points.to(device)
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = model(points, features)
                outputs = outputs.permute(0, 2, 1)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                val_pred.append(preds.detach().cpu())
                val_true.append(labels.detach().cpu())
                
        val_loss /= len(val_loader)
        val_pred = torch.cat(val_pred, dim=0)
        val_true = torch.cat(val_true, dim=0)
        _, val_miou = compute_iou(val_pred, val_true, 9)
        
        print(f"Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}")
        
        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model.")
            
        if args.dry_run:
            print("Dry run completed.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./Toronto_3D/', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size') # Reduced default for safety
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_points', type=int, default=4096, help='Number of points per block')
    parser.add_argument('--block_size', type=float, default=10.0, help='Block size in meters')
    parser.add_argument('--sample_rate', type=float, default=1.0, help='Fraction of data to use')
    parser.add_argument('--dry_run', action='store_true', help='Run for 1 epoch and exit')
    
    args = parser.parse_args()
    train(args)
