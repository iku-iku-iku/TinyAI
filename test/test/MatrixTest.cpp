//
// Created by iku-iku-iku on 2022/10/27.
//

#include "gtest/gtest.h"
#include "Type.h"

TEST(MatrixTest, divide) {
    double data1[1][1] = {{1,},};
    double data2[1][1] = {{2,},};

    Matrix<1, 1> m1(data1);
    Matrix<1, 1> m2(data2);

    std::cout << m1 << m2 << divide_per_elem(m1, m2);
}

TEST(MatrixTest, sqrt) {
    double data2[1][1] = {{2,},};

    Matrix<1, 1> m2(data2);

    std::cout << m2 << sqrt(m2);
}
TEST(MatrixTest, power) {
    double data2[1][1] = {{2,},};

    Matrix<1, 1> m2(data2);

    std::cout << m2 << power_per_elem(m2, 2);
}
