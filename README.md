---
title: "Agglomerative Clustering"
author: "Vladimir Petrov"
---

## Project Overview

This project is a **personal agglomerative clustering** implementation, which forms part of a larger exploration of **Hierarchical Risk Parity**. The purpose of this project is to provide a practical foundation for learning **C++** and **OpenMP** while implementing the clustering algorithm. The algorithm is built to help understand the fundamental concepts of clustering and parallelization. The code doesn't use the most optimized clustering algortihm on purpose, as more loops allow for more implementations of parallelization.

## Clustering Algorithm

The implemented algorithm follows the basic principles of **Agglomerative Hierarchical Clustering**, which is a **bottom-up** approach to cluster analysis. It starts with each data point as its own cluster and iteratively merges the closest clusters based on a chosen linkage method until a stopping criterion is met (e.g., a specific number of clusters).

### Linkage Methods Supported:
- **Single Linkage**: The distance between two clusters is defined as the shortest distance between any pair of points from each cluster.
- **Complete Linkage**: The distance between two clusters is defined as the longest distance between any pair of points from each cluster.
- **Euclidean Linkage**: The distance is calculated using the Euclidean metric, the straight-line distance between points in the feature space.

### Distance Metrics Supported:
- **Euclidean Distance**: The straight-line distance between two points in space.

- **Manhattan Distance**: Also known as the **L1 distance** or **taxicab distance**, which sums the absolute differences between the coordinates of two points.

- **Cosine Similarity**: Measures the cosine of the angle between two vectors, which is used to evaluate the similarity of points based on their direction in the feature space.

## Implementation Details

The project leverages **C++** for implementing the clustering algorithm and **OpenMP** for parallel processing to speed up computations, especially during the calculation of distances and merging of clusters. While the algorithm is not highly optimized, the goal is to build a solid foundation for more complex hierarchical clustering algorithms.

## Algorithm
**Step 1:** The Distances between all the intial points are calculated and kept inside a condensed distance array.

**Step 2:** A map is created for the condensed array in which the indices of the points (clusters) of the corresponding distance are kept.

**For Loop**

- **Step 3:** The minimum distance is found and new cluster is created.

- **Step 4:** All the distances involving the selected points (clusters) are removed from the condensed array and the map, and all remaining ones are shifted to the far left.

- **Step 5:** All distances involving the new cluster are calculated and added to the back of the condensed array and the map is updated as well.

- **Step 6:** Repeat until **n-1** clusters have been found.

## Conclusion

This project is an important step in building a strong understanding of hierarchical clustering and parallelization. It is part of a larger effort to learn **C++** and **OpenMP**, with the aim of applying these techniques in more complex machine learning tasks in the future. Although not the most optimized solution, this implementation serves as a useful foundation for further experimentation and optimization in clustering algorithms.

## Future Work
- Remove the condensed array map.
- Add more methods which return useful information about the clusters.
- Make returns; return a value rather than void and print.
- Have errors in case something goes wrong or wrong input is provided.
- Add Ward linkage.
- Optimize memory allocation of variables

#Dependencies
- C++ 20
- G++ 12.3.0
- OpenMP 201511
