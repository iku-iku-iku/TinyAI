//
// Created by iku-iku-iku on 2022/10/21.
//

#ifndef METAAI_UTIL_H
#define METAAI_UTIL_H

#include <iostream>
#include <random>

inline auto print() {std::cout << std::endl;}
template<typename G, typename... T>
inline auto print(G g, T&&... var) {
    std::cout << g << " ";
    print(var...);
}

template<unsigned seed>
struct NormalDistributionGenerator {
    std::default_random_engine gen;
    std::normal_distribution<double> dis{0, 1};

    NormalDistributionGenerator() : gen(seed) {}

    auto operator()() {
        return dis(gen);
    }
};

struct RandNormalDistributionGenerator {
    std::default_random_engine gen;
    std::normal_distribution<double> dis{0, 1};

    RandNormalDistributionGenerator() {
        gen.seed(std::chrono::system_clock::now().time_since_epoch().count());
    }

    auto operator()() {
        return dis(gen);
    }
};

#endif //METAAI_UTIL_H
