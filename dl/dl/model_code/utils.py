import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_iou(pred, target, num_classes):
    """
    Compute Intersection over Union (IoU) for each class and Mean IoU.
    
    Args:
        pred (torch.Tensor): Predictions (N,).
        target (torch.Tensor): Ground truth labels (N,).
        num_classes (int): Number of classes.
        
    Returns:
        ious (list): List of IoU for each class.
        miou (float): Mean IoU.
    """
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore background/unlabeled if necessary (usually class 0 or -1, but here we assume 0-8 are valid)
    # If there's an ignore index, filter it out here.
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        
        if union == 0:
            ious.append(float('nan'))  # No ground truth for this class
        else:
            ious.append((intersection / union).item())
            
    # Compute mean IoU ignoring NaNs
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    miou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    
    return ious, miou

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx
