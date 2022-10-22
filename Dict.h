//
// Created by iku-iku-iku on 2022/10/20.
//

#ifndef METAAI_DICT_H
#define METAAI_DICT_H

#include <tuple>
#include <memory>

struct A {
};
struct B {
};
struct C {
};

template<bool A, typename B, typename C>
struct condition {
    using type = B;
};

template<typename B, typename C>
struct condition<false, B, C> {
    using type = C;
};

template<typename T1, typename T2>
constexpr auto is_same_type = false;

template<typename T>
constexpr auto is_same_type<T, T> = true;


template<typename... VarKeys>
struct Dict {

    template<typename T>
    static constexpr auto GetIndex() {
        return Find<0, T, VarKeys...>();
    }

    template<std::size_t i, typename T1>
    static constexpr auto Find() {
        return -1;
    }

    template<std::size_t i, typename T1, typename T2, typename... Types>
    static constexpr auto Find() {
        if constexpr (is_same_type<T1, T2>) {
            return i;
        } else {
            return Find<i + 1, T1, Types...>();
        }
    }
    template<typename... Args>
    struct Values {
        template<std::size_t I, typename CurType, typename... Types>
        struct Type {
            using type = typename Type<I - 1, Types...>::type;
        };

        template<typename CurType, typename... Types>
        struct Type<0, CurType, Types...> {
            using type = CurType;
        };

        template<std::size_t I>
        using GetType = typename Type<I, Args...>::type;


        std::shared_ptr<void> m_Tuple[sizeof...(Args)];

        template<typename KeyType, typename ValueType>
        constexpr auto Set(ValueType &&val) {
            constexpr auto index = GetIndex<KeyType>();
            using RawValType = std::decay_t<ValueType>;
            using New = Replace<RawValType, index, Values<>, Args...>;
            auto n = New();
            for (int i = 0; i < sizeof...(Args); i++) {
                if (i != index) { n.m_Tuple[i] = std::move(m_Tuple[i]); }
                else {
                    n.m_Tuple[i] = std::shared_ptr<void>(new RawValType(std::forward<ValueType>(val)), [](void *ptr) {
                        auto p = static_cast<RawValType *>(ptr);
                        delete p;
                    });
                }
            }
            return n;
        }

        template<typename Key>
        [[nodiscard]] auto& Get() const {
            constexpr auto index = GetIndex<Key>();
            using ValType = GetType<index>;
            return *reinterpret_cast<ValType *>(m_Tuple[index].get());
        }

        template<typename NewType, std::size_t I, std::size_t Index, typename PreTypes,
                typename CurType, typename... RemainTypes>
        struct Replace_;

        template<typename NewType, std::size_t I, std::size_t Index, template<typename...> class PreTypes,
                typename... Traveled, typename CurType, typename... RemainTypes>
        struct Replace_<NewType, I, Index, PreTypes<Traveled...>, CurType, RemainTypes...> {
            using type = typename Replace_<NewType, I + 1, Index, PreTypes<Traveled..., CurType>, RemainTypes...>::type;
        };
        template<typename NewType, std::size_t Index, template<typename...> class PreTypes,
                typename... Traveled, typename CurType, typename... RemainTypes>
        struct Replace_<NewType, Index, Index, PreTypes<Traveled...>, CurType, RemainTypes...> {
            using type = PreTypes<Traveled..., NewType, RemainTypes...>;
        };

        template<typename NewType, std::size_t Index, typename PreTypes, typename... RemainTypes>
        using Replace = typename Replace_<NewType, 0, Index, PreTypes, RemainTypes...>::type;

    };

    using Keys = Values<VarKeys...>;

    static constexpr auto Create() { return Keys(); }
};

#endif //METAAI_DICT_H
