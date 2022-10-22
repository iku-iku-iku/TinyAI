//
// Created by iku-iku-iku on 2022/10/21.
//

#ifndef METAAI_TYPE_H
#define METAAI_TYPE_H

#include <cstring>

// 矩阵，可以看成基本类型
template<std::size_t In_N, std::size_t In_M, typename InDataType = double>
struct Matrix {
    using DataType = InDataType;
    static constexpr std::size_t N = In_N;
    static constexpr std::size_t M = In_M;

    DataType data[N][M]{};

    Matrix() = default;

    Matrix(DataType in_data[N][M]) { std::memcpy(data, in_data, sizeof(data)); }

    // 根据传入的随机数生成器生成矩阵
    template<typename Gen>
    static Matrix RandomlyCreate(Gen gen) {
        Matrix res;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                res.data[i][j] = gen();
            }
        }
        return res;
    }

    DataType& get(std::size_t i , std::size_t j = 0) {
        return data[i][j];
    }

    DataType value() { return data; }

    Matrix(const Matrix &) = default;

    Matrix &operator=(const Matrix &) = default;

    void ForwardImpl() {}

    void BackwardImpl() {}

    template<std::size_t R_N, std::size_t R_M>
    Matrix<N, R_M> operator*(const Matrix<R_N, R_M> &rhs) {
        Matrix<N, R_M> res;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < R_M; j++) {
                for (int k = 0; k < R_N; k++) {
                    res.data[i][j] += data[i][k] * rhs.data[k][j];
                }
            }
        }
        return res;
    }

    Matrix operator+(const Matrix &rhs) {
        Matrix res;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                res.data[i][j] = data[i][j] + rhs.data[i][j];
            }
        }
        return res;
    }

    Matrix operator-() {
        Matrix res = *this;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                res.data[i][j] = -data[i][j];
            }
        }
        return res;
    }

    Matrix<M, N> T() {
        Matrix<M, N> res;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                res.data[j][i] = data[i][j];
            }
        }
        return res;
    }

    DataType *operator[](std::size_t i) {
        return data[i];
    }

    Matrix &operator+=(const Matrix &rhs) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                data[i][j] += rhs.data[i][j];
            }
        }
        return *this;
    }

    template<typename T>
    friend Matrix operator*(const Matrix &mat, const T &scalar) {
        Matrix res(mat);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                res.data[i][j] *= scalar;
            }
        }
        return res;
    }

    template<typename T>
    friend Matrix operator*(const T &scalar, const Matrix &mat) {
        return mat * scalar;
    }

    friend auto &operator<<(std::ostream &out, const Matrix &var) {
        std::cout << "data:" << std::endl;
        for (int i = 0; i < N; i++) {
            out << '\t';
            for (int j = 0; j < M; j++) {
                out << var.data[i][j] << ' ';
            }
            out << std::endl;
        }
        return out;
    }
};

#endif //METAAI_TYPE_H
