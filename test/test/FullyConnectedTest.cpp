//
// Created by iku-iku-iku on 2022/10/26.
//

#include <random>
#include <future>
#include "Components.h"
#include "gtest/gtest.h"
#include "Util.h"
#include "Train.h"
#include "Data.h"

void train(auto& seq) {
    Dataset<4> train_dataset;
    train_dataset.read_from_file("../../data/Iris-train.txt");

    AdamOptimizer optimizer(seq.parameters());

    const int epoch = 200;
    for (int e_i = 0; e_i < epoch; ++e_i) {
        double total_loss = 0;
        for (int d_i = 0; d_i < train_dataset.size(); ++d_i) {
            MatrixVariable x{train_dataset.get_feature(d_i)};
            MatrixVariable y{one_hot<3>(train_dataset.get_label(d_i))};

            // 组装
            auto& a2 = seq(x.data);
            auto loss = Sum{((a2 - y) ^ 2) * 0.5};

            loss.Forward();
            optimizer.clear_grad();
            loss.Backward();
            optimizer.step();

            total_loss += loss.whole.value();
        }
    }
}

auto test(auto& seq) {
    Dataset<4> test_dataset;
    test_dataset.read_from_file("../../data/Iris-test.txt");

    int total = 0, right = 0;
    for (int d_i = 0; d_i < test_dataset.size(); ++d_i) {
        MatrixVariable x{test_dataset.get_feature(d_i)};

        // 组装
        auto& pred = seq(x.data);

        total++;
        right += test_dataset.get_label(d_i) == max_i(pred.whole.data);
    }

    return 1.0 * right / total;
}

#define RANDOM

TEST(FullyConnectedTest, test2) {
    std::vector<double> accuracy_vec;
    std::vector<std::future<double>> tasks;
    for (int i = 0; i < 10; i++) {
        Sequence<3, 1,
                FullyConnectedLayer<4, 10, RandNormalDistributionGenerator>,
                SigmoidLayer<10, 1>,
                FullyConnectedLayer<10, 3, RandNormalDistributionGenerator>,
                SigmoidLayer<3, 1>
        > seq;
        train(seq);
        auto accuracy = test(seq);
        accuracy_vec.push_back(accuracy);
    }

    for (auto &fut: tasks) { accuracy_vec.push_back(fut.get()); }
    for (const auto &acc: accuracy_vec) {
        print("accuracy:", acc);
    }

    print("sigma:", sigma(accuracy_vec));
    print("mean:", mean(accuracy_vec));
}
