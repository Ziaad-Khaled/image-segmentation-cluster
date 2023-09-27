# Image Segmentation with Clustering

Welcome to my Image Segmentation with Clustering project! In this mini-project, I've implemented clustering algorithms to perform image segmentation and distinguish between foreground and background objects.

## Overview

This project comprises several tasks, each aimed at enhancing image segmentation:

### 1. Clustering Algorithms

#### 1.1 K-Means Clustering

K-Means clustering is a powerful technique for image segmentation. I've provided an initial implementation in `segmentation.py`, and I've optimized it by leveraging NumPy functions and broadcasting to make it significantly faster. You can find the optimized version as `kmeans_fast`.

#### 1.2 Hierarchical Agglomerative Clustering

Hierarchical Agglomerative Clustering (HAC) is another clustering algorithm I've implemented. It starts with each point in its cluster and iteratively merges clusters until the desired number of clusters is achieved. My implementation can be found in `segmentation.py`.

### 2. Pixel-Level Features

Before applying clustering algorithms, I've computed feature vectors for each pixel, ensuring they encapsulate essential information for segmentation.

#### 2.1 Color Features

In `segmentation.py`, I've implemented `color_features`, which calculates a feature vector based solely on color information for each pixel.

#### 2.2 Color and Position Features

For more informative feature vectors, I've implemented `color_position_features`. This feature vector combines color and position information, enhancing the segmentation quality. Proper normalization is applied to handle variations in feature ranges.

### 3. Quantitative Evaluation

To evaluate the performance of these segmentation algorithms, I've employed a dataset of cat images with corresponding ground-truth segmentations. Using `compute_accuracy` from `segmentation.py`, I quantitatively measure accuracy based on true positives (TP), true negatives (TN), positives (P), and negatives (N).
