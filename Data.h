//
// Created by iku-iku-iku on 2022/10/21.
//

#ifndef METAAI_DATA_H
#define METAAI_DATA_H

#include "Components.h"
#include <fstream>


template<typename T>
inline auto mean(std::vector<T> &vec) {
    T sum{};
    for (const auto &x: vec) {
        sum += x;
    }

    return sum * 1.0 / vec.size();
}

template<typename T>
inline auto deviation(std::vector<T> &vec) {
    T sqrt_sum{};
    for (const auto &x: vec) {
        sqrt_sum += x * x;
    }
    auto mean_value = mean(vec);
    return sqrt_sum * 1.0 / vec.size() - mean_value * mean_value;
}

template<typename T>
inline auto sigma(std::vector<T> &vec) {
    return std::sqrt(deviation(vec));
}

template<std::size_t N, typename ElemType>
auto read_vector(std::stringstream &ss) {
    Vector<N, ElemType> vec;

    for (int i = 0; i < N; i++) {
        ss >> vec.data[i][0];
    }

    return vec;
}

template<typename T>
auto read_elem(std::stringstream &ss) {
    T t;
    ss >> t;
    return t;
}

template<std::size_t N>
auto one_hot(std::size_t i) {
    Vector<N> vec;
    vec.data[i][0] = 1;
    return vec;
}

template<std::size_t N, typename DataType>
auto max_i(Vector<N, DataType> &vec) {
    std::size_t res = -1;
    DataType max_v = -1e9;
    for (std::size_t i = 0; i < N; ++i) {
        if (vec.data[i][0] > max_v) {
            max_v = vec.data[i][0];
            res = i;
        }
    }
    return res;
}

template<std::size_t FeatN, typename InFeatureType = double, typename InLabelType = int>
struct Dataset {
    using FeatureType = InFeatureType;
    using LabelType = InLabelType;

    using VecType = Vector<FeatN, FeatureType>;

    std::vector<VecType> features;

    std::vector<LabelType> labels;

    void read_from_file(const char *path) {
        std::ifstream ifs;

        ifs.open(path, std::ios::in);

        std::stringstream ss;
        char buf[256];
        while (ifs.getline(buf, sizeof(buf))) {
            ss << buf;
            features.push_back(read_vector<FeatN, FeatureType>(ss));
            labels.push_back(read_elem<LabelType>(ss));
        }

        ifs.close();
    }

    void normalize() {
        for (int i = 0; i < FeatN; i++) {
            FeatureType min_v = 1e9, max_v = -1e9;
            for (auto &vec: features) {
                min_v = std::min(min_v, vec.get(i));
                max_v = std::max(max_v, vec.get(i));
            }

            for (auto &vec: features) {
                vec.get(i) = (vec.get(i) - min_v) / (max_v - min_v);
            }
        }
    }


    void normalize2() {
        for (int i = 0; i < FeatN; i++) {
            std::vector<FeatureType> vec;
            for (auto &feat: features) {
                vec.push_back(feat.get(i));
            }

            auto mu = mean(vec);
            auto sig = sigma(vec);

            for (auto &feat: features) {
                feat.get(i) = (feat.get(i) - mu) / sig;
            }
        }
    }

    VecType &get_feature(std::size_t i) { return features[i]; }

    LabelType &get_label(std::size_t i) { return labels[i]; }

    std::size_t size() { return features.size(); }
};

#endif //METAAI_DATA_H
