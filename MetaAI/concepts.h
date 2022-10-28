//
// Created by iku-iku-iku on 2022/10/27.
//

#ifndef METAAI_CONCEPT_H
#define METAAI_CONCEPT_H


#include <type_traits>

// 标量
template<typename T>
concept C_Scalar = std::is_scalar_v<T>;

// MatrixVariable和Matrix都是MatrixLike
// 输出为矩阵的component也可以看成MatrixLike
template<typename T>
concept C_MatrixLike = requires { T::N; T::M; };

// 可优化的类型
template<typename T>
concept C_Optimizable = requires(T t) {
    t.grad;
    t.data;
};

// 单独的变量
template<typename T>
concept C_SingleVariable = requires{ typename T::VariableClassFlag; };

// 矩阵变量
template<typename T>
concept C_MatrixVariable = requires{ typename T::MatrixVariableClassFlag; };

// 变量
template<typename T>
concept C_Variable = C_MatrixVariable<T> || C_SingleVariable<T>;

// 利用Component来构建一棵树
// 将Variable看成一种trivial的component，并且Variable在叶子节点
template<typename T>
concept C_Component = requires(T t) {
    &T::Backward;
    &T::Forward;
    t.whole.data; // 如果是Variable，则为可优化的参数，如果是component，则为中间结果
    t.whole.grad; // 梯度，反向传播时根据component的求导结果和component的输出来计算得出（链式法则）
};

// 矩阵变量，或包含矩阵变量叶子节点的树
template<typename T>
concept C_MatrixLikeComponent = C_Component<T> && C_MatrixLike<T>;

template<typename T>
concept C_ParameterLayer = requires(T t) { t.parameters(); };
#endif //METAAI_CONCEPT_H
