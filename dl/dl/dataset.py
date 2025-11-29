import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pyntcloud import PyntCloud

from sklearn.neighbors import NearestNeighbors

class Toronto3DDataset(Dataset):
    def __init__(self, root_dir, split='train', num_points=4096, block_size=10.0, sample_rate=1.0, transform=None):
        """
        Args:
            root_dir (str): Directory containing the .ply files.
            split (str): 'train', 'val', or 'test'.
            num_points (int): Number of points to sample per block.
            block_size (float): Size of the block for sliding window (meters).
            sample_rate (float): Fraction of blocks to use (for faster training/debugging).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.block_size = block_size
        self.transform = transform
        
        # Toronto-3D splits (L001, L002, L003, L004)
        # Standard split: Train: L001, L003, L004; Val: L002 (or similar, adjusting based on user notebook)
        # Based on notebook: L002 seems to be used for viz, let's assume L002 is Val/Test.
        # Let's define: Train=[L001, L003, L004], Val=[L002] for now.
        if split == 'train':
            self.file_names = ['L001.ply', 'L003.ply', 'L004.ply']
        elif split == 'val':
            self.file_names = ['L002.ply']
        else:
            self.file_names = ['L002.ply'] # Test on L002 for now if not specified

        self.file_paths = [os.path.join(root_dir, f) for f in self.file_names]
        
        self.points = []
        self.labels = []
        self.colors = []
        
        # Load data
        print(f"Loading {split} data from {self.file_paths}...")
        for file_path in self.file_paths:
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} not found.")
                continue
                
            cloud = PyntCloud.from_file(file_path)
            df = cloud.points
            
            # Normalize coordinates (center them roughly to avoid huge numbers)
            # The notebook did: df[['x', 'y', 'z']] -= UTM_OFFSET
            # We will handle local blocks, so global offset matters less for the model, 
            # but good for numerical stability.
            # Let's just use the raw values and block them.
            
            xyz = df[['x', 'y', 'z']].values
            rgb = df[['red', 'green', 'blue']].values / 255.0
            
            if 'scalar_Label' in df.columns:
                lbl = df['scalar_Label'].values
            else:
                lbl = np.zeros(len(df)) # Dummy labels if missing
                
            self.points.append(xyz)
            self.colors.append(rgb)
            self.labels.append(lbl)
            
        # Pre-process into blocks
        self.blocks = []
        self._prepare_blocks()
        
        if sample_rate < 1.0:
            num_blocks = len(self.blocks)
            keep_blocks = int(num_blocks * sample_rate)
            indices = np.random.choice(num_blocks, keep_blocks, replace=False)
            self.blocks = [self.blocks[i] for i in indices]
            
        print(f"Total {split} blocks: {len(self.blocks)}")

    def _prepare_blocks(self):
        for i, points in enumerate(self.points):
            labels = self.labels[i]
            colors = self.colors[i]
            
            # Calculate grid indices for all points at once
            min_xyz = np.min(points, axis=0)
            
            # Floor division to get grid indices
            grid_indices = np.floor((points[:, :2] - min_xyz[:2]) / self.block_size).astype(int)
            
            # Create a dataframe to group by grid indices
            # This is much faster than iterating and masking
            df_block = pd.DataFrame({
                'gx': grid_indices[:, 0],
                'gy': grid_indices[:, 1],
                'idx': range(len(points))
            })
            
            # Group by grid coordinates
            grouped = df_block.groupby(['gx', 'gy'])
            
            for (gx, gy), group in grouped:
                if len(group) < 100:
                    continue
                    
                indices = group['idx'].values
                block_points = points[indices]
                block_labels = labels[indices]
                block_colors = colors[indices]
                
                # Calculate center for this block
                center = min_xyz + np.array([gx * self.block_size + self.block_size/2, 
                                           gy * self.block_size + self.block_size/2, 
                                           0])
                
                self.blocks.append({
                    'points': block_points,
                    'labels': block_labels,
                    'colors': block_colors,
                    'center': center
                })

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        block = self.blocks[idx]
        points = block['points']
        labels = block['labels']
        colors = block['colors']
        
        # Resample to fixed number of points
        if len(points) >= self.num_points:
            choice = np.random.choice(len(points), self.num_points, replace=False)
        else:
            choice = np.random.choice(len(points), self.num_points, replace=True)
            
        points = points[choice]
        labels = labels[choice]
        colors = colors[choice]
        
        # Center points (local coordinates)
        # Keep Z relative to ground? Or just center XYZ?
        # Usually center XY, keep Z absolute or relative to min Z.
        # Let's center XYZ for standard PointNet/RandLA.
        current_points = points - np.mean(points, axis=0)
        
        # Augmentation
        if self.split == 'train':
            # Rotation around Z
            theta = np.random.uniform(0, 2*np.pi)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            current_points = np.dot(current_points, rotation_matrix)
            
            # Scaling
            scale = np.random.uniform(0.9, 1.1)
            current_points *= scale
            
            # Jitter
            current_points += np.random.normal(0, 0.01, current_points.shape)
            
            # Random Flip
            if np.random.random() > 0.5:
                current_points[:, 0] = -current_points[:, 0]
            if np.random.random() > 0.5:
                current_points[:, 1] = -current_points[:, 1]

        # Features: XYZ, RGB, Normalized XYZ
        # RandLA-Net typically uses: XYZ (original), Features (RGB + relative XYZ etc)
        # Here we pass:
        # 1. input_points: (N, 3) - Actual XYZ coords (centered/augmented)
        # 2. input_features: (N, C) - RGB + Intensity (if we had it) + Normalized XYZ
        # 3. labels: (N,)
        
        # Normalized XYZ for features
        min_p = np.min(current_points, axis=0)
        max_p = np.max(current_points, axis=0)
        range_p = max_p - min_p
        # Avoid div by zero
        range_p[range_p == 0] = 1.0
        normalized_xyz = (current_points - min_p) / range_p
        
        # Concatenate features: RGB + Normalized XYZ
        # (N, 3+3) = (N, 6)
        features = np.concatenate((colors, normalized_xyz), axis=1)
        
        return torch.from_numpy(current_points).float(), \
               torch.from_numpy(features).float(), \
               torch.from_numpy(labels).long()
