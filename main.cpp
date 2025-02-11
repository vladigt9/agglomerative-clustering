#include <iostream>
#include <vector>
#include <chrono>

#include <Cluster.h>
#include <Matrix.h>

void testClustering()
{
    // Vector used to debugging and checking results with Scikit-learn implementation
    Matrix<double> X {{
        {2, 5}, {33, 64}, {464, 7454}, {545, 328}, {6, 9546}, {7456, 1210}, {881, 12341}, {899, 16542}, 
        // {2, 5}, {33, 64}, {464, 7454}, {545, 328}, {6, 9546}, {7456, 1210}, {881, 12341}, {899, 16542}, 
        // {2, 5}, {33, 64}, {464, 7454}, {545, 328}, {6, 9546}, {7456, 1210}, {881, 12341}, {899, 16542}, 
        {12180, 1853}, {121, 1844}, {65412, 1325}, {137865, 13216}, {1497, 1752}, {1546, 185}, {16546, 198}, 
        // {12180, 1853}, {121, 1844}, {65412, 1325}, {137865, 13216}, {1497, 1752}, {1546, 185}, {16546, 198}, 
        // {12180, 1853}, {121, 1844}, {65412, 1325}, {137865, 13216}, {1497, 1752}, {1546, 185}, {16546, 198}, 
        // {12180, 1853}, {121, 1844}, {65412, 1325}, {137865, 13216}, {1497, 1752}, {1546, 185}, {16546, 198}, 
        // {12180, 1853}, {121, 1844}, {65412, 1325}, {137865, 13216}, {1497, 1752}, {1546, 185}, {16546, 198}, 
        // {5641, 27820}, {4564, 213}, {876, 45}, {654, 6464}, {21, 24},
        // {5641, 27820}, {4564, 213}, {876, 45}, {654, 6464}, {21, 24},
        // {5641, 27820}, {4564, 213}, {876, 45}, {654, 6464}, {21, 24},
        // {5641, 27820}, {4564, 213}, {876, 45}, {654, 6464}, {21, 24},
        {5641, 27820}, {4564, 213}, {876, 45}, {654, 6464}, {21, 24},},
        20, 2
    };

    Cluster<double> cluster {X, "average", "euclidean", 5, false};

    auto start = std::chrono::high_resolution_clock::now();
    cluster.fit();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Clustering time: " << duration.count() << " milliseconds" << std::endl;
    
    std::vector<int> clusters {cluster.clusters()};
}

// void testMatrix()
// {

//     std::vector<std::vector<double>> x {{1,2,3}, {2,3,4}, {3,4,5}};
//     std::vector<std::vector<double>> y {{1,1,1}, {1,1,1}, {1,1,1}};
//     std::vector<std::vector<double>> z {{2,2,2}, {2,2,2}, {2,2,2}};
//     Matrix<double> A {x, 3, 3};
//     Matrix<double> B {y, 3, 3};
//     Matrix<double> C {z, 3, 3};

    
//     int size = A.getMatrix().size();
    
//     A.add(B);
//     A.subtract(C);
//     A.multiply(C);

//     int m {A.getRows()};
//     int n {A.getColumns()};

//     for (int i = 0; i < m; ++i)
//     {
//         for (int j = 0; j < n; ++j)
//         {
//             std::cout << A[i][j] << ", ";
//         }
//         std::cout << '\n';
//     }
// }

int main()
{
    testClustering();
    // testMatrix();

    return 0;
}
