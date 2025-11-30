import os
import numpy as np
from plyfile import PlyData
import torch
from torch.utils.data import Dataset
import random

CLASS_NAMES = [
    "Unclassified",  # 0
    "Ground",        # 1
    "Road_markings", # 2
    "Natural",       # 3
    "Building",      # 4
    "Utility_line",  # 5
    "Pole",          # 6
    "Car",           # 7
    "Fence",         # 8
]
NUM_CLASSES = len(CLASS_NAMES)

# ==============================================================
# SAFE LOAD FUNCTION (prevents NaN, fixes huge UTM values)
# ==============================================================
def load_ply_points(path, utm_offset=(627285.0, 4841948.0, 0.0), max_points=None):
    ply = PlyData.read(path)
    v = ply["vertex"].data

    # ------------- Coordinates -------------
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    xyz = xyz - np.array(utm_offset, dtype=np.float32)[None, :]   # remove UTM

    # ------------- Features ----------------
    rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32) / 255.0
    intensity = np.asarray(v["scalar_Intensity"], dtype=np.float32)[..., None] / 100.0
    scan_angle = np.asarray(v["scalar_ScanAngleRank"], dtype=np.float32)[..., None] / 20.0
    gps_time = np.asarray(v["scalar_GPSTime"], dtype=np.float32)[..., None] / 300000.0

    feats = np.concatenate([rgb, intensity, scan_angle, gps_time], axis=1)

    labels = np.asarray(v["scalar_Label"], dtype=np.int64)

    # -------- Subsampling --------
    N = xyz.shape[0]
    if max_points is not None and N > max_points:
        idx = np.random.choice(N, size=max_points, replace=False)
        xyz = xyz[idx]
        feats = feats[idx]
        labels = labels[idx]
        print(f"[load_ply_points] {os.path.basename(path)}: subsampled to {max_points} points.")

    # -------- VERY IMPORTANT: remove NaN/infs --------
    xyz = np.nan_to_num(xyz, nan=0.0, posinf=1.0, neginf=-1.0)
    feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=0.0)
    labels = np.nan_to_num(labels, nan=0).astype(np.int64)

    return xyz, feats, labels


# ==============================================================
# SAFE AUGMENTATION (guarantees no NaN)
# ==============================================================
def augment_points(xyz, feats):
    # rotation around Z
    theta = random.uniform(0, 2 * np.pi)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    Rz = np.array([[cos_t, -sin_t, 0.0],
                   [sin_t,  cos_t, 0.0],
                   [0.0,    0.0,   1.0]], dtype=np.float32)
    xyz = xyz @ Rz.T

    # scaling
    xyz *= random.uniform(0.9, 1.1)

    # jitter
    xyz += np.random.normal(0.0, 0.01, size=xyz.shape).astype(np.float32)

    # color jitter
    rgb = feats[:, :3]
    other = feats[:, 3:]
    brightness = random.uniform(0.9, 1.1)
    noise = np.random.normal(0.0, 0.02, size=rgb.shape).astype(np.float32)
    rgb = np.clip(rgb * brightness + noise, 0.0, 1.0)
    feats = np.concatenate([rgb, other], axis=1)

    # CLEAN AGAIN — FINAL SAFETY
    xyz = np.nan_to_num(xyz, nan=0.0, posinf=1.0, neginf=-1.0)
    feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)

    return xyz, feats


# ==============================================================
# TORONTO 3D DATASET CLASS
# ==============================================================
class Toronto3DDataset(Dataset):
    def __init__(self,
                 root="./processed_toronto3d",
                 split="train",
                 num_points=4096,
                 max_points_in_memory=5_000_000,
                 augment=True):
        super().__init__()
        assert split in ["train", "val", "test"]
        self.split = split
        self.num_points = num_points
        self.augment = augment and split == "train"

        ply_path = os.path.join(root, f"{split}.ply")
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"{ply_path} not found.")

        print(f"[Toronto3DDataset] Loading {ply_path}")
        xyz, feats, labels = load_ply_points(ply_path, max_points=max_points_in_memory)

        self.xyz, self.feats, self.labels = xyz, feats, labels
        self.N = xyz.shape[0]
        print(f"[Toronto3DDataset] Split={split}, total points loaded: {self.N}")

        # how many batches per epoch
        self.epoch_size = max(self.N // self.num_points, 100)

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        choice = np.random.choice(self.N, self.num_points, replace=(self.N < self.num_points))

        xyz = self.xyz[choice].copy()
        feats = self.feats[choice].copy()
        labels = self.labels[choice].copy()

        if self.augment:
            xyz, feats = augment_points(xyz, feats)

        # per-sample normalization
        xyz_mean = xyz.mean(axis=0, keepdims=True)
        xyz -= xyz_mean

        xyz_std = xyz.std(axis=0, keepdims=True)
        xyz_std = np.clip(xyz_std, 1e-3, None)   # <--- avoids division by 0
        xyz /= xyz_std

        # FINAL sanity cleaning
        xyz = np.nan_to_num(xyz, nan=0.0, posinf=1.0, neginf=-1.0)
        feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)

        return (torch.from_numpy(xyz).float(),
                torch.from_numpy(feats).float(),
                torch.from_numpy(labels).long())
