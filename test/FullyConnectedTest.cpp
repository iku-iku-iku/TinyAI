//
// Created by iku-iku-iku on 2022/10/26.
//

#include <random>
#include "Components.h"
#include "gtest/gtest.h"
#include "Util.h"

TEST(FullyConnectedTest, test) {
    auto x = MatrixVariable{Matrix<3, 1>::AllOne()};
    auto fc = FullyConnected<3, 5, decltype(x)>{x};
    fc.w.data = Matrix<5, 3>::AllOne();
    fc.b.data = Matrix<5, 1>::AllOne();

    fc.Forward();

    print("x", x.data);
    print("w", fc.w.data);
    print("b", fc.b.data);
    print("out", fc.whole.data);

    x.clear_grad();
    fc.w.clear_grad();
    fc.b.clear_grad();
    fc.Backward();
    print("x grad", x.grad);
    print("w grad", fc.w.grad);
    print("b grad", fc.b.grad);

}