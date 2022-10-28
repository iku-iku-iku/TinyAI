//
// Created by iku-iku-iku on 2022/10/27.
//

#include "gtest/gtest.h"
#include "Components.h"
#include "Util.h"

TEST(LayerTest, test_sequence_with_component) {
    Sequence<4, 1, FullyConnectedLayer<3, 4, decltype([] {return 1; })>, SigmoidLayer<4, 1>> seq;

    auto mat = Matrix<3, 1>::AllOne();
    auto y = MatrixVariable{Matrix<4, 1>::AllOne()};

    auto& res = seq(mat);
    auto diff = res - y;
    auto power = diff ^ 2;
    auto loss = Sum{power};
    loss.Forward();
    loss.Backward();

    ASSERT_EQ(&res, &diff.operand1);
    ASSERT_EQ(&res, &power.operand.operand1);
    ASSERT_EQ(&res, &loss.operand.operand.operand1);
    ASSERT_EQ(&res, &res.whole);
    ASSERT_EQ(loss.operand.operand.operand1.whole.grad, res.grad);
}
