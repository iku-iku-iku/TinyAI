//
// Created by iku-iku-iku on 2022/10/28.
//

#include "gtest/gtest.h"
#include "Util.h"


template<typename...T>
void tuple_checker(std::tuple<T...>);

template<typename T>
concept C_Tuple = requires(T t){ tuple_checker(t); };

template<std::size_t... Is>
auto to_ref_tuple(auto &t, std::index_sequence<Is...>) {
    return std::make_tuple((std::ref(get<Is>(t)), ...));
}

template<typename... Elem>
auto to_ref_tuple(std::tuple<Elem...> &t) {
    return to_ref_tuple(t, std::index_sequence_for<Elem...>{});
}

template<typename... Elem1, typename... Elem2, std::size_t... Is1, std::size_t... Is2>
inline auto
compose(const std::tuple<Elem1...>& t1, const std::tuple<Elem2...>& t2, std::index_sequence<Is1...>, std::index_sequence<Is2...>) {
    return std::make_tuple((get<Is1>(t1), ...), (get<Is2>(t2), ...));
}

template<typename... Elem1, typename... Elem2>
inline auto compose(const std::tuple<Elem1...>& t1, const std::tuple<Elem2...>& t2) {
    return compose(t1, t2, std::index_sequence_for<Elem1...>{}, std::index_sequence_for<Elem2...>{});
}
//
//
//template<typename... Elem>
//inline auto to_flat_tuple(std::tuple<Elem...> t) {
//    return z_to_flat_tuple<0, sizeof...(Elem), Elem...>(t);
//}

TEST(Little, test_to_ref_tuple) {
    auto t = std::make_tuple(1);
    auto x(to_ref_tuple(t));
    get<0>(x) = -1;
    print(get<0>(t));
}

TEST(Little, test2) {
    auto t = std::make_tuple(1, 2);

    static_assert(C_Tuple<decltype(t)>);
    static_assert(!C_Tuple<int>);
}
