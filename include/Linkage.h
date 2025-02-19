#ifndef LINKAGE_H
#define LINKAGE_H


#include <vector>
#include <limits>
#include <memory>

#include "Distance.h"


class Linkage 
{
protected:
    std::unique_ptr<Distance> distanceMetric;

public:
    Linkage(std::unique_ptr<Distance> metric) : distanceMetric(std::move(metric)) {}

    virtual double applyLinkage(const std::vector<double*>& clusterA,
                                const std::vector<double*>& clusterB, int nCols) = 0;

    virtual ~Linkage() {}
};

class AverageLinkage : public Linkage 
{
public:
    AverageLinkage(std::unique_ptr<Distance> metric) 
        : Linkage(std::move(metric)) {}
    
    double applyLinkage(const std::vector<double*>& clusterA,
                        const std::vector<double*>& clusterB, int nCols) override {
        // Calculate the average linkage distance between clusters A and B
        std::size_t mA {clusterA.size()};
        std::size_t mB {clusterB.size()};

        double distance {0};

        // #pragma omp parallel for reduction(+:distance)
        for (int i = 0; i < mA; ++i)
        {
            for (int k = 0; k < mB; ++k)
            {
                std::vector<double> pointA(clusterA[i], clusterA[i] + nCols);
                std::vector<double> pointB(clusterB[k], clusterB[k] + nCols);
                distance += distanceMetric->calculateDistance(pointA, pointB, nCols);
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
};

class SingleLinkage : public Linkage 
{
public:
    SingleLinkage(std::unique_ptr<Distance> metric) 
        : Linkage(std::move(metric)) {}
    
    double applyLinkage(const std::vector<double*>& clusterA,
                        const std::vector<double*>& clusterB, int nCols) override
    {
        std::size_t mA {clusterA.size()};
        std::size_t mB {clusterB.size()};

        // Initialize minDistance with bbig placeholder value
        double minDistance = std::numeric_limits<double>::max();

        // Parallelize the loop using reduciton min so omp does the atomicity
        // #pragma omp parallel for reduction(min : minDistance)
        for (int i = 0; i < mA; ++i)
        {
            for (int j = 0; j < mB; ++j)
            {
                std::vector<double> pointA(clusterA[i], clusterA[i] + nCols);
                std::vector<double> pointB(clusterB[j], clusterB[j] + nCols);
                double distance = distanceMetric->calculateDistance(pointA, pointB, nCols);
                
                if (distance < minDistance)
                {
                    minDistance = distance;
                }
            }
        }

        return minDistance;
    }
};    

class CompleteLinkage : public Linkage 
{
public:
    CompleteLinkage(std::unique_ptr<Distance> metric) 
        : Linkage(std::move(metric)) {}
    
    double applyLinkage(const std::vector<double*>& clusterA,
                        const std::vector<double*>& clusterB, int nCols) override
    {
        std::size_t mA {clusterA.size()};
        std::size_t mB {clusterB.size()};

        // Initialize minDistance with bbig placeholder value
        double maxDistance = std::numeric_limits<double>::min();

        // Parallelize the loop using reduciton min so omp does the atomicity
        // #pragma omp parallel for reduction(max : minDistance)
        for (int i = 0; i < mA; ++i)
        {
            for (int j = 0; j < mB; ++j)
            {
                std::vector<double> pointA(clusterA[i], clusterA[i] + nCols);
                std::vector<double> pointB(clusterB[j], clusterB[j] + nCols);
                double distance = distanceMetric->calculateDistance(pointA, pointB, nCols);

                if (distance > maxDistance)
                {
                    maxDistance = distance;
                }
            }
        }

        return maxDistance;
    }
};    

#endif
