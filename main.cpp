#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <algorithm>
#include <random>
#include <chrono>

// #include <cblas.h>

template <typename T>
class Clustering
{
private:
    // m_X - stores the input matrix
    // m_linkageType - linkage type from "single", "average", "complete"
    // m_metric - distance metric from "euclidean", "manhattan", "cosine"
    // m_clusters - stores the cluster number of each input point
    // m_n_clusters - number of unique clusters which m_cluster stores
    // m_preemptiveStop - if True stops clustering once num of unique clusters = m_n_clusters

    std::vector<std::vector<T>> m_X{};
    std::string m_linkageType{"single"};
    std::string m_metric{"euclidean"};
    std::vector<int> m_clusters {};
    int m_n_clusters {1};
    bool m_preemptiveStop {false};

    // Defines a row for the final linkage matrix
    struct LinkageRow {
        int cluster1 {0};
        int cluster2 {0};
        double distance {0.0};
        int size {0};
    };

    double calculateDistance(const std::vector<T> &a, const std::vector<T> &b)
    {
        // Calculates the distance between two points.

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
        else if (m_metric == "manhattan")
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
        // Averages the position of all the points of two clusters
        // after which returns the distance between them

        // Calculate the centroid of clusterA
        std::vector<double> centroidA(clusterA[0].get().size(), 0.0);

        // Split threads
        // This can be done without localcentroids and using atomic,
        // however must be tested before changing
        #pragma omp parallel
        {
            // Local copy of centroidA for each thread
            std::vector<double> localCentroid(clusterA[0].get().size(), 0.0);

            // Parallize the number of points rather than dimension as usaully
            // for matrix mxn m >> n
            #pragma omp for
            for (std::size_t i = 0; i < clusterA.size(); ++i)
            {
                for (std::size_t k = 0; k < clusterA[i].get().size(); ++k)
                {
                    localCentroid[k] += clusterA[i].get()[k];
                }
            }

            // Combine results from each thread into the global centroid
            #pragma omp critical
            {
                for (std::size_t k = 0; k < centroidA.size(); ++k)
                {
                    centroidA[k] += localCentroid[k];
                }
            }
        }

        // Get cluster size before the loop to offer small increase in speed
        int sizeClusterA {static_cast<int>(clusterA.size())};

        // Average all the values
        #pragma omp parallel for
        for (std::size_t i = 0; i < centroidA.size(); ++i) 
        {
            centroidA[i] /= sizeClusterA;
        }

        // Calculate the centroid of clusterB
        std::vector<double> centroidB(clusterB[0].get().size(), 0.0);


        // Expand the threads
        #pragma omp parallel
        {
            // Local copy of centroidB for each thread
            std::vector<double> localCentroid(clusterB[0].get().size(), 0.0);

            #pragma omp for
            for (std::size_t i = 0; i < clusterB.size(); ++i)
            {
                for (std::size_t k = 0; k < clusterB[i].get().size(); ++k)
                {
                    localCentroid[k] += clusterB[i].get()[k];
                }
            }

            // Combine results
            #pragma omp critical
            {
                for (std::size_t k = 0; k < centroidB.size(); ++k)
                {
                    centroidB[k] += localCentroid[k];
                }
            }
        }

        int sizeClusterB {static_cast<int>(clusterB.size())};

        // Average all the distances
        #pragma omp for
        for (std::size_t i = 0; i < centroidB.size(); ++i) 
        {
            centroidB[i] /= clusterB.size();
        }

        // Find distances between the cntroids of the two clusters
        double distance = calculateDistance(centroidA, centroidB);

        return distance;
    }

    double completeLinkage(const std::vector<std::reference_wrapper<std::vector<double>>>& clusterA,
                     const std::vector<std::reference_wrapper<std::vector<double>>>& clusterB)
    {
        // Initialize with a small value
        double maxDistance = std::numeric_limits<double>::min();

        // Parallelize the for loop and use reduction to find the maxDsitance
        #pragma omp parallel for reduction(max : maxDistance)
        for (std::size_t i = 0; i < clusterA.size(); ++i)
        {
            for (std::size_t j = 0; j < clusterB.size(); ++j)
            {
                double distance = calculateDistance(clusterA[i].get(), clusterB[j].get());
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
        // Initialize minDistance with bbig placeholder value
        double minDistance = std::numeric_limits<double>::max();

        // Parallelize the loop using reduciton min so omp does the atomicity
        #pragma omp parallel for reduction(min : minDistance)
        for (std::size_t i = 0; i < clusterA.size(); ++i)
        {
            for (std::size_t j = 0; j < clusterB.size(); ++j)
            {
                double distance = calculateDistance(clusterA[i].get(), clusterB[j].get());

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
        const std::size_t n {std::size(m_X)};

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
        for (std::size_t i = 0; i < n; ++i)
        {
            id_map[i] = static_cast<int>(i);

            // Break on n-1 as there the last point has no more distances to calculate
            if (i == (n-1)) break;

            for (int j = i + 1; j < n; ++j)
            {
                double distance = calculateDistance(m_X[i], m_X[j]);
                int index = i * n - (i * (i + 1)) / 2 + (j - i - 1);
                condensedArray[index] = distance;
                condensedArrayMap[index] = std::make_pair(i, j);
            }
        }

        for (std::size_t i = 0; i < (n-1); ++i)
        {
            // Find best cluster
            LinkageRow cluster {findBestCluster(condensedArray, condensedArrayMap)};

            // Variable to save the size of the cluster
            int size {0};

            // Update id_map and get cluster size
            #pragma omp parallel for reduction(+:size)
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

            // update the clusters array with the id_map when specific point is reached
            if (static_cast<int>(i) == (n-1-m_n_clusters))
            {
                m_clusters = id_map;
            }

            // Break loop when Z has n-1 rows as no more clusters are possible
            // Or break loop when desired number of cluster is reached and preemptive stop is on
            if (i == (n-2) || (static_cast<int>(i) == (n-1-m_n_clusters) && m_preemptiveStop))
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
            std::vector<std::reference_wrapper<std::vector<double>>> cluster1 {};

            for (std::size_t k = 0; k < n; ++k)
            {
                if (id_map[k] == n + i)
                {
                    cluster1.push_back(std::ref(m_X[k]));
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

                condensedArray[numberValidDistances+k] = Linkage(cluster1, cluster2);
            }
            
            // Clear memory
            cluster1.clear();

        }

        return Z;
    }

public:
    Clustering(std::vector<std::vector<T>> X, std::string_view linkage_type, std::string_view metric, int n_clusters, bool preemptiveStop)
        : m_X{X}, m_linkageType{linkage_type}, m_metric{metric}, m_n_clusters{n_clusters}, m_preemptiveStop{preemptiveStop}
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
    

int main()
{
    // Vector used to debugging and checking results with Scikit-learn implementation
    std::vector<std::vector<double>> X {
        {2, 5}, {33, 64}, {464, 7454}, {545, 328}, {6, 9546}, {7456, 1210}, {881, 12341}, {899, 16542}, 
        {12180, 1853}, {121, 1844}, {65412, 1325}, {137865, 13216}, {1497, 1752}, {1546, 185}, {16546, 198}, 
        {5641, 27820}, {4564, 213}, {876, 45}, {654, 6464}, {21, 24}
    };

    Clustering<double> cluster {X, "single", "euclidean", 5, false};

    auto start = std::chrono::high_resolution_clock::now();
    cluster.fit();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Clustering time: " << duration.count() << " milliseconds" << std::endl;
    
    std::vector<int> clusters {cluster.clusters()};

    return 0;
}
