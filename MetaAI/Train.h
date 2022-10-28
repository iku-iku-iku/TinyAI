//
// Created by iku-iku-iku on 2022/10/21.
//

#ifndef METAAI_TRAIN_H
#define METAAI_TRAIN_H

#include <tuple>
#include "Components.h"
#include <vector>

struct LrScheduler {
    double lr;
    double decay_rate;

    explicit LrScheduler(double init_lr = 0.005, double decay_rate_ = 0.9999) : lr(init_lr), decay_rate(decay_rate_) {}

    double get() const { return lr; }

    double next() {
        lr *= decay_rate;
        return lr;
    }
};

template<C_Optimizable... Param>
struct AdamOptimizer {
    static constexpr double beta1 = 0.9;
    static constexpr double beta2 = 0.999;
    static constexpr double eps = 1e-8;

    double beta1_t = beta1;
    double beta2_t = beta2;

    std::vector<Variable *> parameters;

    LrScheduler scheduler{0.0009, 0.9999};

    std::tuple<Param &...> params_to_optimize;

    std::tuple<data_type_of_t<Param>...> v{};
    std::tuple<data_type_of_t<Param>...> S{};

    explicit AdamOptimizer(Param &... params) : params_to_optimize(params...) {}
    explicit AdamOptimizer(std::tuple<Param &...> params) : params_to_optimize(params) {}

    template<std::size_t... Is>
    void step(double lr, std::index_sequence<Is...>) {
        ((get<Is>(v) = beta1 * get<Is>(v) + (1 - beta1) * get<Is>(params_to_optimize).grad), ...);
        ((get<Is>(S) = beta2 * get<Is>(S) + (1 - beta2) * power_per_elem(get<Is>(params_to_optimize).grad, 2)
        ), ...);

        ((get<Is>(params_to_optimize).data += -lr * divide_per_elem(
                (get<Is>(v) / (1 - beta1_t)),
                (eps + sqrt(get<Is>(S) / (1 - beta2_t)))
        )
        ), ...);

        beta1_t *= beta1;
        beta2_t *= beta2;
    }

    template<std::size_t... Is>
    void clear_grad(std::index_sequence<Is...>) {
        (get<Is>(params_to_optimize).clear_grad(), ...);
    }

    void step(double lr) {
        step(lr, std::index_sequence_for<Param...>{});
    }

    void step() {
        step(scheduler.next());
    }

    void clear_grad() {
        clear_grad(std::index_sequence_for<Param...>{});
    }
};

template<C_Optimizable... Param>
struct Optimizer {

    std::vector<Variable *> parameters;

    LrScheduler scheduler{0.1, 0.9999};

    std::tuple<Param &...> params_to_optimize;

    explicit Optimizer(Param &... params) : params_to_optimize(params...) {}
    explicit Optimizer(std::tuple<Param &...> params) : params_to_optimize(params) {}

    template<std::size_t... Is>
    void step(double lr, std::index_sequence<Is...>) {
        ((get<Is>(params_to_optimize).data += -lr * get<Is>(params_to_optimize).grad
        ), ...);
    }

    template<std::size_t... Is>
    void clear_grad(std::index_sequence<Is...>) {
        (get<Is>(params_to_optimize).clear_grad(), ...);
    }

    void step(double lr) {
        step(lr, std::index_sequence_for<Param...>{});
    }

    void step() {
        step(scheduler.next());
    }

    void clear_grad() {
        clear_grad(std::index_sequence_for<Param...>{});
    }
};

#endif //METAAI_TRAIN_H
