#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <limits>
#include <omp.h>

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
    std::string m_linkageType{"single"};
    std::string m_metric{"euclidean"};
    std::vector<int> m_clusters {};
    std::pair<int, int> m_size {1,1};
    int m_n_clusters {1};
    bool m_preemptiveStop {false};

    // Defines a row for the final linkage matrix
    struct LinkageRow {
        int cluster1 {0};
        int cluster2 {0};
        double distance {0.0};
        int size {0};
    };

    double calculateDistance(const std::vector<double>& a, const std::vector<double>& b)
    {
        // Calculates the distance between two points.

        if (m_metric == "euclidean")
        {
            // Initialize variable sum of the squared distances SSD
            double sum = 0.0;

            // Use openMP to parallelize the loop
            #pragma omp parallel for reduction(+ : sum)
            for (size_t i = 0; i < m_X.getColumns(); ++i)
            {
                sum += (a[i] - b[i]) * (a[i] - b[i]);
            }

            // Get euclidean distance by taking the square root
            double distance = std::sqrt(sum);

            return distance;
        }
        else if (m_metric == "manhattan")
        {
            // Initialize variable for final distance
            double distance = 0;

            // Use openMP to parallelize the loop
            #pragma omp parallel for reduction(+ : distance)
            for (size_t i = 0; i < m_X.getRows(); i++)
            {
                distance += std::abs(a[i] - b[i]);
            }

            return distance;
        }
        else if (m_metric == "cosine")
        {
            // Initialize varaibles of the dot products and the norms of the vectors
            double dotProduct = 0;
            double normA = 0, normB = 0;

            // Use openMP to parallelize the loop
            #pragma omp parallel for reduction(+ : dotProduct) reduction(+ : normA) reduction(+ : normB)
            for (size_t i = 0; i < m_X.getRows(); i++)
            {
                dotProduct += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }

            normA = std::sqrt(normA);
            normB = std::sqrt(normB);

            // Calculate cosine similarity
            double cosineSimilarity = dotProduct / (normA * normB);

            // Return cosine distance
            return 1 - cosineSimilarity;
        }

        return 0.0;
    }

    double averageLinkage(Matrix<double>& clusterA,
                     Matrix<double>& clusterB)
    {
        // Averages the distances of all points between 2 cluster
        // after which returns the distance

        int mA {clusterA.getRows()};
        int mB {clusterB.getRows()};

        double distance {0};

        #pragma omp parallel for reduction(+:distance)
        for (int i = 0; i < mA; ++i)
        {
            for (int k = 0; k < mB; ++k)
            {
                distance += calculateDistance(clusterA.vector(i), clusterB.vector(k));
            }
        }

        if (mA < 2 && mB <2)
        {
        }
        else if (mA < 2)
        {
            distance /= mB;
        }
        else if (mB < 2)
        {
            distance /= mA;
        }
        else
        {
            distance /= (mA+mB);
        }

        return distance;
    }

    double completeLinkage(Matrix<double>& clusterA,
                     Matrix<double>& clusterB)
    {
        int mA {clusterA.getRows()};
        int mB {clusterB.getRows()};
        int n {m_size.second};

        // Initialize with a small value
        double maxDistance = std::numeric_limits<double>::min();

        // Parallelize the for loop and use reduction to find the maxDsitance
        #pragma omp parallel for reduction(max : maxDistance)
        for (int i = 0; i < mA; ++i)
        {
            for (int j = 0; j < mB; ++j)
            {
                double distance = calculateDistance(clusterA.vector(i), clusterB.vector(j));
                if (distance > maxDistance)
                {
                    maxDistance = distance;
                }
            }
        }

        return maxDistance;
    }

    double singleLinkage(Matrix<double>& clusterA,
                     Matrix<double>& clusterB)
    {
        int mA {clusterA.getRows()};
        int mB {clusterB.getRows()};
        int n {clusterA.getColumns()};

        // Initialize minDistance with bbig placeholder value
        double minDistance = std::numeric_limits<double>::max();

        // Parallelize the loop using reduciton min so omp does the atomicity
        // #pragma omp parallel for reduction(min : minDistance)
        for (int i = 0; i < mA; ++i)
        {
            for (int j = 0; j < mB; ++j)
            {
                double distance = calculateDistance(clusterA.vector(i), clusterB.vector(j));

                if (distance < minDistance)
                {
                    minDistance = distance;
                }
            }
        }

        return minDistance;
    }

    double Linkage(Matrix<double>& clusterA,
                     Matrix<double>& clusterB)
    {
        // Invokes one of the 3 used linkage methods and returns the distance
        // between the two clusters inputed
        if (m_linkageType == "single")
        {
            return singleLinkage(clusterA, clusterB);
        }
        else if (m_linkageType == "complete")
        {
            return completeLinkage(clusterA, clusterB);
        }
        else if (m_linkageType == "average")
        {
            return averageLinkage(clusterA, clusterB);
        }

        // To be replaced with an error
        return 0.0;
    }

    std::pair<std::vector<double>, std::vector<std::pair<int, int>>> 
    updateCondensedArrayAndMap(const std::vector<double>& condensedArray, 
                               const std::vector<std::pair<int, int>>& arrayMap, 
                               const std::vector<int>& remainingIndices, 
                               int cluster1, 
                               int cluster2) 
    {
        // Updates the condesed array and map by removing all the instances,
        // of the points/clusters which construct the new one then shifts the 
        // remaining values to the left
        
        // Get the new cluster from the back of remainingIndices
        const int newCluster = remainingIndices.back();
        const int sizeIndices = remainingIndices.size();

        // Initialize the new array and map in which all the distances and indices will be stored
        std::vector<double> newCondensedArray {};
        std::vector<std::pair<int, int>> newArrayMap {};

        // Reserve space based on size
        newCondensedArray.reserve((sizeIndices * (sizeIndices - 1)) / 2);
        newArrayMap.reserve((sizeIndices * (sizeIndices - 1)) / 2);

        // Iterate through arrayMap and keep only the valid indices
        for (size_t i = 0; i < arrayMap.size(); ++i) 
        {
            const std::pair<int, int>& pair = arrayMap[i];
            
            // Skip pairs involving merged clusters
            if (pair.first == cluster1 || pair.first == cluster2 || 
                pair.second == cluster1 || pair.second == cluster2)
                continue;

            // Keep the distance and indices that corresponds to this pair
            newArrayMap.push_back(pair);
            newCondensedArray.push_back(condensedArray[i]);
        }

        // Append new distances involving the newly formed cluster
        for (int i = 0; i < static_cast<int>(remainingIndices.size()) - 1; ++i) 
        {
            int a = remainingIndices[i];
            newArrayMap.emplace_back(a, newCluster);
            newCondensedArray.push_back(std::numeric_limits<double>::max()); 
        }

        return {newCondensedArray, newArrayMap};
    }


    std::vector<int> getRemainingIndices(const std::vector<int>& id_map, 
                                     int cluster1, int cluster2) 
    {
        // Returns a vector with the unique remaining indices in ascending order.

        // Initialize the vector and reserve enough space
        std::vector<int> remainingIndices;

        // Add unique elements from id_map to remainingIndices, excluding cluster1 and cluster2
        for (int index : id_map) {
            if (index != cluster1 && index != cluster2) {
                // Only add unique indices
                if (std::find(remainingIndices.begin(), remainingIndices.end(), index) == remainingIndices.end()) {
                    remainingIndices.push_back(index);
                }
            }
        }

        // Sort the remaining indices
        std::sort(remainingIndices.begin(), remainingIndices.end());

        return remainingIndices;
    }

    LinkageRow findBestCluster(std::vector<double>& condensedArray, std::vector<std::pair<int, int>>& condensedArrayMap)
    {
        // Finds the min distance in an array and its index
        // Returns a LinkageRow with cluster indices, the distance and dummy variable for num clusters

        double minDistance = std::numeric_limits<double>::max();
        std::size_t minIndex = 0;

        // Find the index of the minimum distance and the distance
        for (std::size_t i = 0; i < condensedArray.size(); ++i) {
            if (condensedArray[i] < minDistance) {
                minDistance = condensedArray[i];
                minIndex = i;
            }
        }

        return LinkageRow {condensedArrayMap[minIndex].first, condensedArrayMap[minIndex].second, minDistance, 0};
    }

    std::vector<LinkageRow> cluster()
    {
        // Perform the main clustering
        // First calculates distances for the intial matrix
        // Then starts the loop:
        //      * Get best cluster
        //      * Update condensed array and map
        //      * Update Distances
        // Returns the linkage matrix

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

        // Initialize the condensed distance array in memory and its map
        int condensedSize = n * (n - 1) / 2;
        std::vector<double> condensedArray(condensedSize);
        std::vector<std::pair<int, int>> condensedArrayMap (condensedSize);
        
        // Initialize map used to deduce intial cluster from index and current cluster from value
        std::vector<int> id_map (n);

        // Get intial distance condensed array
        for (int i = 0; i < n; ++i)
        {
            id_map[i] = i;

            // Break on n-1 as there the last point has no more distances to calculate
            if (i == (n-1)) break;

            for (int j = i + 1; j < n; ++j)
            {
                std::vector<double> pointA {m_X.vector(i)};
                std::vector<double> pointB {m_X.vector(j)};

                double distance = calculateDistance(pointA, pointB);
                int index = i * n - (i * (i + 1)) / 2 + (j - i - 1);
                condensedArray[static_cast<std::size_t>(index)] = distance;
                condensedArrayMap[static_cast<std::size_t>(index)] = std::make_pair(i, j);
            }
        }

        for (int i = 0; i < (n-1); ++i)
        {
            // Find best cluster
            LinkageRow cluster {findBestCluster(condensedArray, condensedArrayMap)};

            // Variable to save the size of the cluster
            int size {0};

            // Update id_map and get cluster size
            #pragma omp parallel for reduction(+:size)
            for (int k = 0; k < n; ++k)
            {
                if (id_map[static_cast<std::size_t>(k)] == cluster.cluster1 || 
                    id_map[static_cast<std::size_t>(k)] == cluster.cluster2)
                {
                    id_map[static_cast<std::size_t>(k)] = n+i;
                    ++size;
                }
            }

            // Update Linkage matrix
            cluster.size = size;
            Z[static_cast<std::size_t>(i)] = cluster;

            // update the clusters array with the id_map when specific point is reached
            if (i == (n-1-m_n_clusters))
            {
                m_clusters = id_map;
            }

            // Break loop when Z has n-1 rows as no more clusters are possible
            // Or break loop when desired number of cluster is reached and preemptive stop is on
            if (i == (n-2) || (i == (n-1-m_n_clusters) && m_preemptiveStop))
                break;

            // Get remaining indeces of clusters
            std::vector<int> remainingIndices {getRemainingIndices(id_map, cluster.cluster1, cluster.cluster2)};

            // Get the number of valid distances left after the merging
            int validClusters = remainingIndices.size() - 2;
            int numberValidDistances = (validClusters * (validClusters - 1)) / 2 + validClusters;

            // Update the condensed array and map
            std::pair<std::vector<double>, std::vector<std::pair<int, int>>> updateCondensed 
                {updateCondensedArrayAndMap(condensedArray, condensedArrayMap, 
                remainingIndices, cluster.cluster1, cluster.cluster2)};
            
            condensedArray = updateCondensed.first;
            condensedArrayMap = updateCondensed.second;
            
            // Initialize vector to store references to the the data points of each cluster
            // std::vector<std::reference_wrapper<std::vector<double>>> cluster1 {};
            // Matrix<double> cluster1 {};
            Matrix<double> cluster1 {{}, 0, 2};

            for (int k = 0; k < n; ++k)
            {
                if (id_map[static_cast<std::size_t>(k)] == n + i)
                {
                    cluster1.push_row(m_X.vector(k));
                }
            }

            // Update distances
            for (std::size_t k = 0; k < remainingIndices.size()-1; ++k)
            {
                // Initialize a vector to keep references to data points of cluster 2
                Matrix<double> cluster2 {{}, 0, 2};

                for (int l = 0; l < n; ++l)
                {
                    if (id_map[static_cast<std::size_t>(l)] == remainingIndices[static_cast<std::size_t>(k)])
                    {
                        cluster2.push_row(m_X.vector(l));
                    }
                }

                condensedArray[numberValidDistances+k] = Linkage(cluster1, cluster2);
            }
        }

        return Z;
    }

public:
    Cluster(Matrix<T>& X, std::string_view linkage_type, std::string_view metric, int n_clusters, bool preemptiveStop)
        : m_X{X}, m_linkageType{linkage_type}, m_metric{metric},  m_size{X.getRows(), X.getColumns()}, m_n_clusters{n_clusters}, m_preemptiveStop{preemptiveStop}
    {
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
