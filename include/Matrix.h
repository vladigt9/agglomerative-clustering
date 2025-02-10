#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <cblas.h>
#include <utility>
#include <stdexcept>

template <typename T>
class Matrix
{
private:
    std::vector<T> m_X;
    int m_m {1};
    int m_n {1};
    int m_size {1};

    class RowProxy
    {
    private:
        T* row_data;
        int cols;
    public:
        RowProxy(T* data, int cols) : row_data(data), cols(cols) {}

        T& operator[](int col)
        {
            if (col < 0 || col >= cols)
                throw std::out_of_range("Column index out of bounds");
            return row_data[col];
        }

        const T& operator[](int col) const
        {
            if (col < 0 || col >= cols)
                throw std::out_of_range("Column index out of bounds");
            return row_data[col];
        }
    };

public:
    // Constructor to flatten the 2D vector
    Matrix(const std::vector<std::vector<T>>& X, int m, int n)
        : m_m{m}, m_n{n}, m_size{m*n} 
    {
        m_X.reserve(m * n);
        for (const auto& row : X)
            m_X.insert(m_X.end(), row.begin(), row.end());
    }

    RowProxy operator[](int row)
    {
        if (row < 0 || row >= m_m)
            throw std::out_of_range("Row index out of bounds");
        return RowProxy(&m_X[row * m_n], m_n);
    }

    std::pair<int, int> getShape()
    {
        return {m_m, m_n};
    }

    int getRows()
    {
        return m_m;
    }

    int getColumns()
    {
        return m_n;
    }

    std::vector<T> getMatrix()
    {
        return m_X;
    }

    T* getData()
    {
        return m_X.data();
    }

    void add(Matrix<T>& y)
    {
        cblas_daxpy(m_size, 1.0, y.getData(), 1, m_X.data(), 1);
    }
    
    void subtract(Matrix<T>& y)
    {
        cblas_daxpy(m_size, -1.0, y.getData(), 1, m_X.data(), 1);
    }

    void multiply(Matrix<T>& y)
    {
        int p {y.getColumns()};
        std::vector<T> C (m_m*y.getColumns());

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
            m_m, y.getColumns(), m_n, 1.0, m_X.data(), m_n, y.getData(), y.getColumns(), 0.0, C.data(), p);
            
        m_X = std::move(C);

        m_X.shrink_to_fit();
    }
};

#endif
