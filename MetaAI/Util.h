//
// Created by iku-iku-iku on 2022/10/21.
//

#ifndef METAAI_UTIL_H
#define METAAI_UTIL_H

#include <iostream>

inline auto print() {std::cout << std::endl;}
template<typename G, typename... T>
inline auto print(G g, T&&... var) {
    std::cout << g << " ";
    print(var...);
}

#endif //METAAI_UTIL_H
