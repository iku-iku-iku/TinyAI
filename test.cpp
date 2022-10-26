//
// Created by iku-iku-iku on 2022/10/21.
//
#include "MetaAI/Components.h"
#include "Util.h"

void test1() {
    auto v1 = Variable(1);
    auto v2 = Variable(2);
    auto v3 = Variable(3);
    auto v4 = Variable(4);
    auto v5 = Variable(5);
    auto v6 = Variable(6);
    auto v7 = Variable(7);

    VarMultiply layer2{v5, v6, v7};

    VarAddition layer3{v1, v2, v3, v4, layer2};
    layer3.Forward();
    layer3.Backward();

    print(v1.gradient(), v2.gradient(), v3.gradient(), v4.gradient(), v5.gradient(), v6.gradient(), v7.gradient());
}

void test2() {
    double mat1[2][2] = {{2, 0},
                         {1, 2}};
    double mat2[2][2] = {{1, 0},
                         {3, 0}};
    Matrix<2, 2> m1{mat1};
    Matrix<2, 2> m2{mat2};

    MatrixVariable mv1{m1};
    MatrixVariable mv2{m2};

    MatAddition layer{mv1, mv2};

    layer.Forward();
    layer.Backward();
    std::cout << mv2.grad;
}

