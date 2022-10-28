//
// Created by iku-iku-iku on 2022/10/26.
//

#include <future>
#include "Components.h"
#include "gtest/gtest.h"
#include "Data.h"
#include "Util.h"
#include "Train.h"


#define MV(NAME, N, M, g) \
auto NAME##_mat = Matrix<N, M>::CreateWith(g); \
MatrixVariable NAME{NAME##_mat};                                  \
//


void train(auto &w1, auto &b1, auto &w2, auto &b2) {
    Dataset<4> train_dataset;
    train_dataset.read_from_file("../../data/Iris-train.txt");
    Optimizer optimizer{w1, b1, w2, b2};


    const int epoch = 500;
    for (int e_i = 0; e_i < epoch; ++e_i) {
        double total_loss = 0;
        for (int d_i = 0; d_i < train_dataset.size(); ++d_i) {
            MatrixVariable x{train_dataset.get_feature(d_i)};
            MatrixVariable y{one_hot<3>(train_dataset.get_label(d_i))};

            // 组装
            auto a1 = Sigmoid{w1 * x + b1};

            auto a2 = Sigmoid{w2 * a1 + b2};

            auto loss = Sum{((a2 - y) ^ 2) * 0.5};

            loss.Forward();

            optimizer.clear_grad();
            loss.Backward();
            optimizer.step();

            total_loss += loss.whole.value();
        }

    }
}

auto test(auto &w1, auto &b1, auto &w2, auto &b2) {
    Dataset<4> test_dataset;
    test_dataset.read_from_file("../../data/Iris-test.txt");

    int total = 0, right = 0;
    for (int d_i = 0; d_i < test_dataset.size(); ++d_i) {
        MatrixVariable x{test_dataset.get_feature(d_i)};

        // 组装
        auto a1 = Sigmoid{w1 * x + b1};

        auto pred = Sigmoid{w2 * a1 + b2};

        pred.Forward();

        total++;
        right += test_dataset.get_label(d_i) == max_i(pred.whole.data);
    }

    return 1.0 * right / total;
}

#define RANDOM

TEST(BaseClassifierTest, test) {
    std::vector<double> accuracy_vec;
    std::vector<std::future<double>> tasks;
    for (int i = 0; i < 10; i++) {
        tasks.push_back(std::async([] {
#ifdef RANDOM
                                       auto g = RandNormalDistributionGenerator{};
#else
                                       auto g = NormalDistributionGenerator<0>{};
#endif

                                       MV(w1, 10, 4, g)
                                       MV(w2, 3, 10, g)
                                       MV(b1, 10, 1, g)
                                       MV(b2, 3, 1, g)

                                       train(w1, b1, w2, b2);
                                       auto accuracy = test(w1, b1, w2, b2);
                                       return accuracy;
                                   }
        ));
    }

    for (auto &fut: tasks) { accuracy_vec.push_back(fut.get()); }
    for (const auto &acc: accuracy_vec) {
        ASSERT_GT(acc, 0.9);
        print("accuracy:", acc);
    }

    print("sigma:", sigma(accuracy_vec));
    print("mean:", mean(accuracy_vec));
}