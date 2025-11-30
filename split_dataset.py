import os
import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.model_selection import train_test_split

DATA_DIR = "./Toronto_3D"
OUTPUT_DIR = "./processed_toronto3d"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read and merge all .ply files
all_points = []
for file in os.listdir(DATA_DIR):
    if file.endswith(".ply"):
        path = os.path.join(DATA_DIR, file)
        plydata = PlyData.read(path)
        vertex = plydata['vertex'].data
        arr = np.array([tuple(v) for v in vertex], dtype=vertex.dtype)
        all_points.append(arr)
        print(f"Loaded: {file}, {arr.shape[0]} points")

all_points = np.concatenate(all_points, axis=0)
print(f"\nTotal points merged: {all_points.shape[0]}")

# Split dataset (points, NOT file names)
train_data, temp_data = train_test_split(all_points, test_size=0.4, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print("\nSplits:")
print(f"Train: {train_data.shape[0]}")
print(f"Val  : {val_data.shape[0]}")
print(f"Test : {test_data.shape[0]}")

def save_ply(data, filename):
    el = PlyElement.describe(data, 'vertex')
    PlyData([el]).write(filename)

save_ply(train_data, os.path.join(OUTPUT_DIR, "train.ply"))
save_ply(val_data, os.path.join(OUTPUT_DIR, "val.ply"))
save_ply(test_data, os.path.join(OUTPUT_DIR, "test.ply"))

print("\n✔ Saved processed data to:", OUTPUT_DIR)
