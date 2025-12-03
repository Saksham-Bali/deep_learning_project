# Semantic Segmentation of Outdoor 3D Point Clouds

### Enhancing RandLA-Net with Dynamic Graph Convolutions and Density-Aware Sampling

## 1. Introduction and Summary

3D semantic segmentation aims to assign a semantic label to every point in a LiDAR point cloud, enabling scene understanding for autonomous driving, mapping, and robotics. Unlike images, point clouds are unstructured and vary in density, making conventional CNNs unsuitable. Our project focuses on improving an efficient baseline architecture, **RandLA-Net**, by addressing two major issues identified through experimentation:

1. imbalance between dense and sparse regions, and
2. performance degradation on thin or small objects.

To solve these, we introduce two architectural modifications:

* **Dynamic Graph Convolutions** to replace static geometric neighbors in deeper layers, and
* **Density-Aware Sampling** to correct the sampling bias caused by uneven LiDAR density.

These modifications improve segmentation performance on sparse and small objects (poles, fences, utility lines), at the cost of a small drop in the performance of large, easy classes such as road and building.

## 2. Problem Statement

Outdoor LiDAR datasets such as Toronto3D exhibit irregular point density, class imbalance, and a wide variety of object scales. 

* Dense points close to the sensor are oversampled by random sampling.
* Sparse points from far-range structures (utility lines, poles) are underrepresented.
* Small or thin objects suffer significant IoU drop due to neighborhood contamination from large background structures when using fixed geometric kNN.

Our goal was to improve the segmentation quality on such underperforming classes while maintaining overall efficiency.


## 3. Dataset Information: Toronto-3D

The **Toronto-3D** dataset is a large-scale outdoor LiDAR benchmark captured using the **Teledyne Optech Maverick** Mobile LiDAR System.

### Key Characteristics

* Approximately **78 million points** over a **1 km urban roadway**.
* Rich real-world complexity with varying point density.
* Eight semantic classes:
  Road, Road Marking, Natural, Building, Utility Line, Pole, Car, Fence.

### Why Toronto3D?

* Earlier experiments on small single-scene datasets (Oakland) plateaued at low mIoU (~20%).
* Toronto3D’s multi-scene, high-density structure supports better generalization and reveals architecture-level strengths and weaknesses, making it ideal for evaluating sampling and neighborhood modeling strategies.


## 4. Evaluation Metrics

Accuracy proved inadequate for segmentation because it is dominated by large classes (roads, buildings).
Therefore, our evaluation relies on:

### Primary Metric

* **Mean Intersection over Union (mIoU)**
  Measures overlap between predicted and true regions for each class.

### Secondary Metrics

* **Per-Class IoU**
* **Overall Accuracy (OA)**
* **Qualitative Error Maps** 

mIoU provides a reliable assessment of performance across both large and small object classes.


## 5. Baseline Model: RandLA-Net

The baseline model is **RandLA-Net**, an efficient architecture for large point clouds.

### 5.1 Core Architectural Components

#### a) Random Sampling

RandLA-Net downsamples points layer-by-layer using **uniform random sampling**, which is efficient but biases training toward dense regions.

#### b) Local Feature Aggregation (LFA)

Each layer performs two-stage aggregation:

1. **Local Spatial Encoding (LocSE)** – captures relative position, Euclidean offsets.
2. **Attentive Pooling** – aggregates neighbor features using learned attention.

#### c) Encoder–Decoder Structure

* **5 encoder layers** progressively reduce point count.
* **5 decoder layers** reconstruct original resolution via nearest-neighbor interpolation.
* Skip connections fuse geometric detail with semantic features.

### 5.2 Limitations Observed

Based on experimentation and visualizations:

* Random sampling oversamples dense regions and undersamples sparse, informative regions.
* Fixed geometric kNN mixes points from unrelated classes, especially for thin objects.
* LFA pools contaminated features, leading to large drops in IoU for poles, fences, and utility lines.

These observations motivated our improvements.

## 6. Proposed Architectural Improvements

### Overview

Our enhancements directly address the failure modes observed on Toronto3D:

* **Distorted neighborhood context** for thin objects 
* **Sampling imbalance** leading to sparse-region underrepresentation 

### 6.1 Dynamic Graph Convolutions (DGCNN-Inspired)

#### Problem

Fixed XYZ-based kNN frequently mixes points from small objects with dominant background surfaces.
This contaminates local features, causing the model to misclassify small object classes.

#### Solution

Starting from encoder layer 3, we compute **dynamic kNN in the learned feature space** instead of XYZ space.

* **Layers 0–1:** geometric kNN (stable local structure)
* **Layers 2–4:** dynamic feature-space kNN (semantic relationships)

Implementation insight from slide 24: compute pairwise feature distances and rebuild the neighborhood graph per layer. 

#### Effect

* Points belonging to the same semantic object cluster together even if geometrically apart.
* Contamination from large background surfaces is reduced.
* Improves mIoU for sparse and thin classes.

### 6.2 Density-Aware Sampling

#### Problem

Raw random sampling overrepresents dense regions near the sensor and neglects sparse distant regions. 

#### Solution

We introduce a **density-based weighting** mechanism during sampling.

Steps:

1. Compute local density via mean kNN distance.
2. Normalize density across the cloud.
3. Use inverse density as the sampling probability weight.

* Dense regions → lower sampling probability
* Sparse regions → higher sampling probability

#### Effect

* Reduces redundant dense-area samples.
* Maintains representation of underrepresented sparse areas (utility lines, poles).
* Produces more balanced and informative batches.



## 7. Performance Analysis

### 7.1 Positive Outcomes

* **Improved IoU for underperforming classes** such as Utility Line, Pole, Fence.
* Dynamic feature-space neighborhoods resolved contamination issues.
* Density normalization improved learning on sparse long-range structures.

### 7.2 Trade-offs

* Small IoU drop on large continuous classes (Road, Building):
  *Dynamic kNN adds slight noise in regions where static geometry is already optimal.*
* Overall mIoU slightly lower than paper SOTA:
  **73.89% vs 75.3%**  .

However, the model became more robust across classes and better balanced.


## 9. Conclusion

This project demonstrated that improving **local neighborhood quality** and **sampling fairness** leads to significantly better performance on challenging small and sparse object classes in urban LiDAR segmentation.

By integrating **Dynamic Graph Convolutions** and **Density-Aware Sampling**, we addressed the two primary weaknesses of RandLA-Net.
Although the global mIoU saw a small reduction compared to SOTA, the improvements on critical underrepresented classes validated the effectiveness of our architectural choices.

