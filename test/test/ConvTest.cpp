//
// Created by iku-iku-iku on 2022/10/26.
//

#include <random>
#include "Components.h"
#include "gtest/gtest.h"
#include "Util.h"

auto inline RandomGenerator(unsigned seed) {
    std::default_random_engine gen(seed);
    std::normal_distribution<double> dis(0, 1);
    return std::make_tuple(dis, gen);
}

TEST(TestConv, test) {
    auto mv1 = MatrixVariable{Matrix<5, 5>::AllOne()};

    print("mat", mv1.data);

    auto conv = Conv2d<3, 1, decltype(mv1)>{mv1};
    conv.initialize(Matrix<3, 3>::AllOne());
    print("filter", conv.parameters().data);

    conv.Forward();
    print("out", conv.whole.data);

    conv.parameters().clear_grad();
    mv1.clear_grad();
    conv.Backward();
    print("filter grad", conv.parameters().grad);
    print("input grad", mv1.grad);
}