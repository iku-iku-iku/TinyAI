//
// Created by iku-iku-iku on 2022/10/21.
//

#ifndef METAAI_TRAIN_H
#define METAAI_TRAIN_H

#include <tuple>
#include "Components.h"

struct LrScheduler {
    double lr;
    double decay_rate;

    LrScheduler(double init_lr = 0.985, double decay_rate_ = 0.985) : lr(init_lr), decay_rate(decay_rate_) {}

    double get() { return lr; }

    double next() { lr *= decay_rate; return lr;}
};

template<Component Model, typename... Param>
struct Plan {
    LrScheduler scheduler;

    Model model;

    std::tuple<Param &...> param_tuple;


    Plan(LrScheduler scheduler_, Model model_, Param &... param) : scheduler(scheduler_), model(model_), param_tuple(param...) {}

    template<std::size_t... Is>
    void optimize(std::index_sequence<Is...>) {
        auto lr = scheduler.next();
        (get<Is>(param_tuple).step(lr), ...);
        (get<Is>(param_tuple).clear_grad(), ...);
    }

    void step() {
        model.Forward();
        model.Backward();
        optimize(std::index_sequence_for<Param...>{});
    }
};

#endif //METAAI_TRAIN_H
