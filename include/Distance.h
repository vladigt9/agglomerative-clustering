#ifndef DISTANCE_H
#define DISTANCE_H


#include <vector>
#include <limits>
#include <math.h>

class Distance {
public:
    // Virtual function that will be overridden by derived classes
    virtual double calculateDistance(const std::vector<double>& pointA,
        const std::vector<double>& pointB, int nCols
    ) = 0;

    virtual ~Distance() {}
};

class EuclideanDistance : public Distance {
public:
    double calculateDistance(const std::vector<double>& pointA, const std::vector<double>& pointB, int nCols) override {
        // Initialize variable sum of the squared distances SSD
        double sum = 0.0;

        // Use openMP to parallelize the loop
        // #pragma omp parallel for reduction(+ : sum)
        for (size_t i = 0; i < nCols; ++i)
        {
            sum += (pointA[i] - pointB[i]) * (pointA[i] - pointB[i]);
        }

        // Get euclidean distance by taking the square root
        double distance = std::sqrt(sum);

        return distance;

    }
};

class ManhattanDistance : public Distance {
public:
    double calculateDistance(const std::vector<double>& pointA, const std::vector<double>& pointB, int nCols) override {
        double distance = 0;

        // Use openMP to parallelize the loop
        // #pragma omp parallel for reduction(+ : distance)
        for (size_t i = 0; i < nCols; i++)
        {
            distance += std::abs(pointA[i] - pointB[i]);
        }

        return distance;

    }
};

class CosineDistance : public Distance {
public:
    double calculateDistance(const std::vector<double>& pointA, const std::vector<double>& pointB, int nCols) override {
        double dotProduct = 0;
        double normA = 0, normB = 0;

        // Use openMP to parallelize the loop
        // #pragma omp parallel for reduction(+ : dotProduct) reduction(+ : normA) reduction(+ : normB)
        for (size_t i = 0; i < nCols; i++)
        {
            dotProduct += pointA[i] * pointB[i];
            normA += pointA[i] * pointA[i];
            normB += pointB[i] * pointB[i];
        }

        normA = std::sqrt(normA);
        normB = std::sqrt(normB);

        // Calculate cosine similarity
        double cosineSimilarity = dotProduct / (normA * normB);

        // Return cosine distance
        return 1 - cosineSimilarity;
    }
};

#endif
