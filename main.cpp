#include <iostream>
#include <vector>
#include <cmath>
#include <cblas.h>
#include <omp.h>
#include <algorithm>

template <typename T>
class Clustering
{
private:
    std::vector<std::vector<T>> m_X{};
    std::string m_linkageType{"single"};
    std::string m_metric{"euclidean"};

    struct LinkageRow {
        int cluster1 {0};
        int cluster2 {0};
        double distance {0.0};
        int size {0};
    };

    double calculateDistance(const std::vector<T> &a, const std::vector<T> &b)
    {
        if (m_metric == "euclidean")
        {
            // Initialize variable sum of the squared distances SSD
            double sum = 0.0;

            // Use openMP to parallelize the loop
            #pragma omp parallel for reduction(+ : sum)
            for (size_t i = 0; i < a.size(); ++i)
            {
                sum += (a[i] - b[i]) * (a[i] - b[i]);
            }

            // Get euclidean distance by taking the square root
            double distance = std::sqrt(sum);

            return distance;
        }
        else if (m_metric == "mahattan")
        {
            // Initialize variable for final distance
            double distance = 0;

            // Use openMP to parallelize the loop
            #pragma omp parallel for reduction(+ : distance)
            for (size_t i = 0; i < a.size(); i++)
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
            for (size_t i = 0; i < a.size(); i++)
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

    double averageLinkage(const std::vector<std::reference_wrapper<std::vector<double>>>& clusterA,
                     const std::vector<std::reference_wrapper<std::vector<double>>>& clusterB)
    {
        // Calculate the centroid of clusterA
        std::vector<double> centroidA(clusterA[0].get().size(), 0.0);  // Assuming all points have the same dimensions

        for (const auto& pointA : clusterA) 
        {
            for (size_t i = 0; i < pointA.get().size(); ++i) 
            {
                centroidA[i] += pointA.get()[i];
            }
        }

        for (double& val : centroidA) 
        {
            val /= clusterA.size();  // Average the values to get the centroid
        }

        // Calculate the centroid of clusterB
        std::vector<double> centroidB(clusterB[0].get().size(), 0.0);

        for (const auto& pointB : clusterB) 
        {
            for (size_t i = 0; i < pointB.get().size(); ++i) 
            {
                centroidB[i] += pointB.get()[i];
            }
        }

        for (double& val : centroidB) 
        {
            val /= clusterB.size();  // Average the values to get the centroid
        }

        double distance = calculateDistance(centroidA, centroidB);

        return distance;
    }

    double completeLinkage(const std::vector<std::reference_wrapper<std::vector<double>>>& clusterA,
                     const std::vector<std::reference_wrapper<std::vector<double>>>& clusterB)
    {
        double maxDistance = std::numeric_limits<double>::min(); // Initialize with a small value

        // Iterate over each point in clusterA
        for (const auto& pointA : clusterA)
        {
            // Iterate over each point in clusterB
            for (const auto& pointB : clusterB)
            {
                // Calculate the distance between pointA and pointB
                // double distance = calculateDistance(pointA, pointB);
                double distance = calculateDistance(pointA.get(), pointB.get());

                // Update minDistance if a smaller distance is found
                if (distance > maxDistance)
                {
                    maxDistance = distance;
                }
            }
        }

        return maxDistance;
    }

    double singleLinkage(const std::vector<std::reference_wrapper<std::vector<double>>>& clusterA,
                     const std::vector<std::reference_wrapper<std::vector<double>>>& clusterB)
    {
        double minDistance = std::numeric_limits<double>::max(); // Initialize with a large value

        // Iterate over each point in clusterA
        for (const auto& pointA : clusterA)
        {
            // Iterate over each point in clusterB
            for (const auto& pointB : clusterB)
            {
                // Calculate the distance between pointA and pointB
                // double distance = calculateDistance(pointA, pointB);
                double distance = calculateDistance(pointA.get(), pointB.get());

                // Update minDistance if a smaller distance is found
                if (distance < minDistance)
                {
                    minDistance = distance;
                }
            }
        }

        return minDistance;
    }

    double Linkage(const std::vector<std::reference_wrapper<std::vector<double>>>& clusterA,
                     const std::vector<std::reference_wrapper<std::vector<double>>>& clusterB)
    {
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
                                    int cluster1, int cluster2) 
        {
        int newCluster = remainingIndices.back();  // New cluster is at the back of remainingIndices
        std::vector<double> newCondensedArray;
        std::vector<std::pair<int, int>> newArrayMap;

        // Reserve space based on the expected new size
        newCondensedArray.reserve((remainingIndices.size() * (remainingIndices.size() - 1)) / 2);
        newArrayMap.reserve((remainingIndices.size() * (remainingIndices.size() - 1)) / 2);

        // Iterate through `arrayMap` and keep only the valid indices
        for (size_t i = 0; i < arrayMap.size(); ++i) 
        {
            const auto& pair = arrayMap[i];
            
            // Skip pairs involving merged clusters
            if (pair.first == cluster1 || pair.first == cluster2 || 
                pair.second == cluster1 || pair.second == cluster2)
                continue;

            // Keep the distance that corresponds to this pair
            newArrayMap.push_back(pair);
            newCondensedArray.push_back(condensedArray[i]);  // Preserve same indexing
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
        double minDistance = std::numeric_limits<double>::max();
        int minIndex = -1;

        // Find the index of the minimum distance
        for (std::size_t i = 0; i < condensedArray.size(); ++i) {
            if (condensedArray[i] < minDistance) {
                minDistance = condensedArray[i];
                minIndex = i;
            }
        }

        return LinkageRow {condensedArrayMap[minIndex].first, condensedArrayMap[minIndex].second, minDistance, 1};
    }

    std::vector<LinkageRow> clustering()
    {
        // Get initial size of the arr
        auto n {std::size(m_X)};

        // Initialize the linkage matrix
        std::vector<LinkageRow> Z (n-1);

        // Initialize the condensed distance array in memory.
        int condensedSize = n * (n - 1) / 2;
        std::vector<double> condensedArray(condensedSize, std::numeric_limits<double>::max());
        
        // Initialize map used to deduce intial cluster from index and current cluster from value
        std::vector<int> id_map (n);

        // Initialize map used to deduce indices of clusters
        std::vector<std::pair<int, int>> condensedArrayMap {};
        // condensedArrayMap.reserve(n * (n - 1) / 2);

        // Get intial distance condensed array
        for (int i = 0; i < n; ++i)
        {
            id_map[i] = static_cast<int>(i);

            if (i == (n-1)) break;

            for (int j = i + 1; j < n; ++j)
            {
                condensedArrayMap.push_back(std::make_pair(i, j));
                double distance = calculateDistance(m_X[i], m_X[j]);
                int index = i * n - (i * (i + 1)) / 2 + (j - i - 1);
                condensedArray[index] = distance;
            }
        }

        for (std::size_t i = 0; i < (n-1); ++i)
        {
            // Find first best cluster
            LinkageRow cluster {findBestCluster(condensedArray, condensedArrayMap)};

            // Variable to save the size of the cluster
            int size {};

            // Update ID_map
            for (std::size_t k = 0; k < n; ++k)
            {
                if (id_map[k] == cluster.cluster1 || id_map[k] == cluster.cluster2)
                {
                    id_map[k] = n+i;
                    ++size;
                }
            }

            // Update Linkage matrix
            cluster.size = size;
            Z[i] = cluster;

            // Get remaining indeces of clusters
            std::vector<int> remainingIndices {getRemainingIndices(id_map, cluster.cluster1, cluster.cluster2)};

            // Get 
            int validClusters = remainingIndices.size() - 2;
            int newSize = (validClusters * (validClusters - 1)) / 2 + validClusters;

            // Update the condensed array and map
            std::pair<std::vector<double>, std::vector<std::pair<int, int>>> updateCondensed 
                {updateCondensedArrayAndMap(condensedArray, condensedArrayMap, 
                remainingIndices, cluster.cluster1, cluster.cluster2)};
            
            condensedArray = updateCondensed.first;
            condensedArrayMap = updateCondensed.second;
            
            // Initialize vector to store references to the the data points of each cluster
            std::vector<std::reference_wrapper<std::vector<double>>> cluster1 {};

            for (std::size_t k = 0; k < n; ++k)
            {
                if (id_map[k] == n+i)
                {
                    cluster1.push_back(m_X[k]);
                }
            }

            // Update distances
            for (std::size_t k = 0; k < remainingIndices.size()-1; ++k)
            {
                // Initialize a vector to keep references to data points of cluster 2
                std::vector<std::reference_wrapper<std::vector<double>>> cluster2 {};

                for (std::size_t l = 0; l < n; ++l)
                {
                    if (id_map[l] == remainingIndices[k])
                    {
                        cluster2.push_back(m_X[l]);
                    }
                }

                condensedArray[newSize+k] = Linkage(cluster1, cluster2);
            }

        }

        return Z;
    }

public:
    Clustering(std::vector<std::vector<T>> X, std::string_view linkage_type, std::string_view metric)
        : m_X{X}, m_linkageType{linkage_type}, m_metric{metric}
    {
    }

    void fit()
    {
        std::vector<LinkageRow> linkageMap {clustering()};

        for (const auto& row : linkageMap) {
            std::cout << "Cluster1: " << row.cluster1
                    << ", Cluster2: " << row.cluster2
                    << ", Distance: " << row.distance
                    << ", Size: " << row.size
                    << std::endl;
        }

    }
};

int main()
{
    std::vector<std::vector<double>> X{
        {1, 1},
        {1, 2},
        {5, 5},
        {5, 6},
        {15, 15}};

    Clustering<double> cluster {X, "average", "euclidean"};

    cluster.fit();

    return 0;
}
