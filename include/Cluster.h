#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <limits>
#include <numeric>

#include <omp.h>

#include <Linkage.h>
#include <Distance.h>
#include <Matrix.h>  

template <typename T>
class Cluster
{
private:
    // m_X - stores the input matrix
    // m_linkageType - linkage type from "single", "average", "complete"
    // m_metric - distance metric from "euclidean", "manhattan", "cosine"
    // m_clusters - stores the cluster number of each input point
    // m_n_clusters - number of unique clusters which m_cluster stores
    // m_preemptiveStop - if True stops clustering once num of unique clusters = m_n_clusters

    Matrix<T> m_X {};
    std::vector<double> m_average_distances {};
    std::string m_linkageType{"single"};
    std::string m_metric{"euclidean"};
    std::vector<int> m_clusters {};
    std::pair<int, int> m_size {1,1};
    int m_n_clusters {1};
    bool m_preemptiveStop {false};
    std::unique_ptr<Linkage> linkageStrategy;
    std::unique_ptr<Distance> distanceStrategy;
    std::unique_ptr<Distance> distanceStrategy2;

    // Defines a row for the final linkage matrix
    struct LinkageRow {
        int cluster1 {0};
        int cluster2 {0};
        double distance {0.0};
        int size {0};
    };

    double performLinkage(std::vector<double*>& clusterA, std::vector<double*>& clusterB)
    {
        return linkageStrategy->applyLinkage(clusterA, clusterB, m_X.getRows());  // Polymorphism in action
    }

    LinkageRow getBestCluster(std::vector<double>& condensedArray,
        std::vector<int> idMap, int n) {
        double minDistance = std::numeric_limits<double>::max();
        std::size_t minIndex = 0;

        // Find the index of the minimum distance and the distance
        for (std::size_t i = 0; i < condensedArray.size(); ++i) {
            if (condensedArray[i] < minDistance) {
                minDistance = condensedArray[i];
                minIndex = i;
            }
        }

        std::pair<int, int> clusterIndices {getClusterPairFromIndex(minIndex, n)};
        
        int i {idMap[clusterIndices.first]};
        int j {idMap[clusterIndices.second]};

        if (i > j) {
            std::swap(i, j);
        }

        return LinkageRow {i, j, minDistance, 0};
    }

    void updateClusterMap(std::vector<int>& clusterMap, LinkageRow& cluster, int clusterN)
    {
       // Update cluster_map
        for (std::size_t i = 0; i < clusterMap.size(); ++i)
        {
            if (clusterMap[i] == cluster.cluster1 || 
                clusterMap[i] == cluster.cluster2)
            {
                clusterMap[i] = clusterN;
            }
        }
    }

    void updateIdMap(std::vector<int>& idMap, LinkageRow& cluster, int clusterN)
    {
        for (std::size_t k = 0; k < idMap.size(); ++k)
        {
            if (idMap[k] == cluster.cluster1)
            {
                idMap[k] = -1;
            }
            else if (idMap[k] == cluster.cluster2)
            {
                idMap[k] = clusterN;
            }
        }
    }
    
    int getClusterSize(std::vector<int>& clusterMap, int newClusterN)
    {
        int size {0};
        
        // #pragma omp parallel for reduction(+:size)
        for (std::size_t i = 0; i < clusterMap.size(); ++i)
        {
            if (clusterMap[i] == newClusterN)
            {
                ++size;
            }
        }
        return size;
    }

    std::pair<int, int> getClusterPairFromIndex(int index, int n) {
        // Find the row `i` in the upper triangular matrix
        
        int b = 1 - (2 * n);
        
        auto ss = (-b - sqrt(b*b - 8 * index)) / 2;

        int i = static_cast<int>((-b - sqrt(b*b - 8 * index)) / 2);
    
        // Now calculate `j` based on the formula
        int j = index + i * (b + i + 2) / 2 + 1;

        return std::pair<int, int> {i,j};
    }

    int getIndexFromClusterPair(int i, int j, int n) {
        // Ensure that i < j, if not, swap them
        if (i > j) {
            std::swap(i, j);
        }
    
        // Calculate the index using the formula
        return i * n - (i * (i + 1)) / 2 + (j - i - 1);
    }

    std::vector<double*> getCluster1(
        std::vector<int>& clusterMap, std::vector<int>& idMap,
        int currentN, int n
    )
    {
    
        std::vector<double*> c1 {};

        for (std::size_t k = 0; k < clusterMap.size(); ++k)
        {
            if (clusterMap[k] == currentN)
            {
                c1.push_back(m_X.getRowAddress(k));
            }
        }

        return c1;
    }

    int getIndex(const std::vector<int>& vec, int value) {
        auto it = std::find(vec.begin(), vec.end(), value);
        if (it != vec.end()) {
            return std::distance(vec.begin(), it); // Compute index
        }
        return -1; // Return -1 if value is not found
    }

    void updateDistArray(
        std::vector<double>& condensedDistArray, std::vector<int>& idMap,
        std::vector<int>& clusterMap, LinkageRow cluster, int currentN, int n
    )
    {   
        for (std::size_t i = 0; i < condensedDistArray.size(); ++i)
        {
            for (std::size_t j = i + 1; j < condensedDistArray.size(); ++j)
            {
                if (i == cluster.cluster1 || j == cluster.cluster1)
                {
                    int index = getIndexFromClusterPair(i,j,n);
                    condensedDistArray[index] = std::numeric_limits<double>::max();
                }
            }
        }

        std::vector<double*> cluster1 = getCluster1(clusterMap, idMap, currentN, n);

        int c1Index {getIndex(idMap, currentN)};

        for (std::size_t i = 0; i < idMap.size(); ++i)
        {
            if (idMap[i] == -1 || idMap[i] == currentN)
                continue;
                
            std::vector<double*> cluster2 {};

            for (std::size_t j = 0; j<clusterMap.size(); ++j)
            {
                if (idMap[i] == clusterMap[j])
                {
                    cluster2.push_back(m_X.getRowAddress(j));
                }
            }

            if (cluster2.size() == 0)
                continue;
            
            double distance = performLinkage(cluster1, cluster2);
            
            int index {getIndexFromClusterPair(i, c1Index, n)};
            
            condensedDistArray[index] = distance;
        }
    }

    std::vector<LinkageRow> cluster()
    {
        // Get initial size of the arr
        const int n {m_X.getRows()};

        // Initialize the linkage matrix and resize based on return number of clusters
        std::vector<LinkageRow> Z {};
        if (m_preemptiveStop && m_n_clusters > 0)
        {
            Z.resize(n - m_n_clusters, LinkageRow());
        }
        else
        {
            Z.resize(n - 1, LinkageRow());
        }

        std::vector<int> idMap (n);
        std::iota(idMap.begin(), idMap.end(), 0);

        std::vector<int> clusterMap (n);
        std::iota(clusterMap.begin(), clusterMap.end(), 0);

        // Initialize the condensed distance array in memory and its map
        int initialCondensedSize = n * (n - 1) / 2;
        std::vector<double> condensedDistArray(initialCondensedSize);

        // Get intial distance condensed array
        for (int i = 0; i < n-1; ++i)
        {
            for (int j = i + 1; j < n; ++j)
            {
                std::vector<double> pointA {m_X.vector(i)};
                std::vector<double> pointB {m_X.vector(j)};

                double distance = distanceStrategy2->calculateDistance(pointA, pointB, n);
                int index = getIndexFromClusterPair(i, j, n);
                condensedDistArray[static_cast<std::size_t>(index)] = distance;
            }
        }

        for (int i = 0; i < (n-1); ++i)
        {
            // Find best cluster
            int newClusterN {n+i};
            LinkageRow cluster {getBestCluster(condensedDistArray, idMap, n)};
            updateClusterMap(clusterMap, cluster, newClusterN);
            cluster.size = getClusterSize(clusterMap, newClusterN);
            Z[static_cast<std::size_t>(i)] = cluster;

            if ((n-2) == i)
                break;

            updateIdMap(idMap, cluster, newClusterN);
            
            updateDistArray(condensedDistArray, idMap, clusterMap, cluster, newClusterN, n);
        }
    
        return Z;
    }

public:
    Cluster(Matrix<T>& X, std::string_view linkage_type, std::string_view metric, int n_clusters, bool preemptiveStop)
        : m_X{X}, m_linkageType{linkage_type}, m_metric{metric},  m_size{X.getRows(), X.getColumns()}, m_n_clusters{n_clusters}, m_preemptiveStop{preemptiveStop}
    {

        if (m_metric == "euclidean") {
            distanceStrategy = std::make_unique<EuclideanDistance>();
            distanceStrategy2 = std::make_unique<EuclideanDistance>();
        } 
        else if (m_metric == "manhattan") {
            distanceStrategy = std::make_unique<ManhattanDistance>();
            distanceStrategy2 = std::make_unique<ManhattanDistance>();
        } 
        else {
            distanceStrategy = std::make_unique<CosineDistance>();
            distanceStrategy2 = std::make_unique<CosineDistance>();
        }
        

        // Move the distanceStrategy into linkageStrategy only once
        if (m_linkageType == "single") {
            linkageStrategy = std::make_unique<SingleLinkage>(std::move(distanceStrategy));
        } 
        else if (m_linkageType == "complete") {
            linkageStrategy = std::make_unique<CompleteLinkage>(std::move(distanceStrategy));
        } 
        else {
            linkageStrategy = std::make_unique<AverageLinkage>(std::move(distanceStrategy));
        }

        // Ensure linkageStrategy is valid after moving
        if (!linkageStrategy) {
            throw std::runtime_error("Error: linkageStrategy is nullptr after move!");
        }

    }

    void fit()
    {
        // Will return a std::vector<double> Linkage matrix
        // Currently left as void for faster debugging and less code

        std::vector<LinkageRow> linkageMap {cluster()};

        // Print for debugging
        for (const auto& row : linkageMap) {
            std::cout << "Cluster1: " << row.cluster1
                    << ", Cluster2: " << row.cluster2
                    << ", Distance: " << row.distance
                    << ", Size: " << row.size
                    << std::endl;
        }

    }

    std::vector<int> clusters()
    {
        // Returns a vector with the the cluster number for each point from the input matrix

        // Print for debugging
        for (int i : m_clusters)
        {
            std::cout << i << ", ";
        }
        std::cout << '\n';

        return m_clusters;
    }
};


#endif // CLUSTERING_H
