---
title: "Agglomerative Clustering"
author: "Vladimir Petrov"
---

## Project Overview

This project is a **personal agglomerative clustering** implementation, which forms part of a larger exploration of **Hierarchical Clustering**. The purpose of this project is to provide a practical foundation for learning **C++** and **OpenMP** while implementing the clustering algorithm. The algorithm is built to help understand the fundamental concepts of clustering, and although it is not the most optimized solution, it serves as a great starting point for further optimization and learning.

## Clustering Algorithm

The implemented algorithm follows the basic principles of **Agglomerative Hierarchical Clustering**, which is a **bottom-up** approach to cluster analysis. It starts with each data point as its own cluster and iteratively merges the closest clusters based on a chosen linkage method until a stopping criterion is met (e.g., a specific number of clusters).

### Linkage Methods Supported:
- **Single Linkage**: The distance between two clusters is defined as the shortest distance between any pair of points from each cluster.
- **Complete Linkage**: The distance between two clusters is defined as the longest distance between any pair of points from each cluster.
- **Euclidean Linkage**: The distance is calculated using the Euclidean metric, the straight-line distance between points in the feature space.

### Distance Metrics Supported:
- **Euclidean Distance**: The straight-line distance between two points in space, which is calculated as:
  
  \[
  d(p, q) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2}
  \]
  
- **Manhattan Distance**: Also known as the **L1 distance** or **taxicab distance**, which sums the absolute differences between the coordinates of two points:
  
  \[
  d(p, q) = \sum_{i=1}^{n} |p_i - q_i|
  \]

- **Cosine Similarity**: Measures the cosine of the angle between two vectors, which is used to evaluate the similarity of points based on their direction in the feature space.

## Implementation Details

The project leverages **C++** for implementing the clustering algorithm and **OpenMP** for parallel processing to speed up computations, especially during the calculation of distances and merging of clusters. While the algorithm is not highly optimized, the goal is to build a solid foundation for more complex hierarchical clustering algorithms.

### Optimization Considerations

- The algorithm is implemented in a simple, straightforward manner for clarity and learning purposes.
- Future optimizations could include improving the distance calculation efficiency, optimizing memory usage, and experimenting with more efficient linkage criteria.
- Further exploration with **parallelization using OpenMP** could also help reduce runtime for larger datasets.

## Conclusion

This project is an important step in building a strong understanding of hierarchical clustering. It is part of a larger effort to learn **C++** and **OpenMP**, with the aim of applying these techniques in more complex machine learning tasks in the future. Although not the most optimized solution, this implementation serves as a useful foundation for further experimentation and optimization in clustering algorithms.

