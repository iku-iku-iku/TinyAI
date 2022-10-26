//
// Created by iku-iku-iku on 2022/10/21.
//

#ifndef METAAI_COMPONENTS_H
#define METAAI_COMPONENTS_H

#include <iostream>
#include "Type.h"
#include <tuple>
#include <cstring>
#include <cmath>

// 定义单操作数的MatrixLikeComponent的N和M
#define SINGLE_OPERAND_MATRIX_NM  \
static constexpr std::size_t N = InOperand::N; \
static constexpr std::size_t M = InOperand::M; \
//

// 标量
template<typename T>
concept Scalar = std::is_scalar_v<T>;

// MatrixVariable和Matrix都是MatrixLike
// 输出为矩阵的component也可以看成MatrixLike
template<typename T>
concept MatrixLike = requires { T::N; T::M; };

// 可优化的类型
template<typename T>
concept Optimizable = requires(T t) {
    &T::step;
    &T::clear_grad;
};

// 单独的变量
template<typename T>
concept IsSingleVariable = requires{ typename T::VariableClassFlag; };

// 矩阵变量
template<typename T>
concept IsMatrixVariable = requires{ typename T::MatrixVariableClassFlag; };

// 变量
template<typename T>
concept IsVariable = IsMatrixVariable<T> || IsSingleVariable<T>;

// 利用Component来构建一棵树
// 将Variable看成一种trivial的component，并且Variable在叶子节点
template<typename T>
concept Component = requires(T t) {
    &T::Backward;
    &T::Forward;
    t.whole.data; // 如果是Variable，则为可优化的参数，如果是component，则为中间结果
    t.whole.grad; // 梯度，反向传播时根据component的求导结果和component的输出来计算得出（链式法则）
};

template<typename T>
concept MatrixLikeComponent = Component<T> && MatrixLike<T>;

template<typename InDataType>
struct Variable;

template<std::size_t InN, std::size_t InM>
struct MatrixVariable;

template<typename OperandType>
struct z_operand_ref {
    using type = std::remove_reference_t<OperandType>;
};

// 目前是完全在栈上运算，效率更高，因此需要对于栈上的Variable需要引用
template<typename InDataType>
struct z_operand_ref<Variable<InDataType>> {
    using type = std::remove_reference_t<Variable<InDataType>> &;
};

template<std::size_t InN, std::size_t InM>
struct z_operand_ref<MatrixVariable<InN, InM>> {
    using type = std::remove_reference_t<MatrixVariable<InN, InM>> &;
};

template<typename InOperandType>
using operand_ref_t = typename z_operand_ref<InOperandType>::type;


// 单独的变量
template<typename InDataType>
struct Variable {

    using VariableClassFlag = std::nullptr_t;

    using DataType = InDataType;

    DataType data{};
    DataType grad{1}; // 对自己的梯度为1

    Variable &whole;

    Variable() : whole(*this) {}

    explicit Variable(DataType x) : data(x), whole(*this) {}

    friend auto &operator<<(std::ostream &out, const Variable &var) {
        return out << var.data;
    }

    DataType value() { return data; }

    DataType gradient() { return grad; }

//    Variable(const Variable &) = delete;
//
//    Variable &operator=(const Variable &) = delete;

    // 利用梯度更新参数
    void step(double lr) {
        data += -lr * grad;
    }

    // 清除梯度
    void clear_grad() {
        grad = DataType{};
    }

    void Forward() {}

    void Backward() {}
};

template<typename... Operands>
struct VarMultiply {

    using Whole = Variable<double>;

    std::tuple<Operands &...> operands_tuple;

    Whole whole{}; // 看成一个整体

    explicit VarMultiply(Operands &... operands) : operands_tuple(operands...) {}

    template<std::size_t... Is>
    auto VarForward(std::index_sequence<Is...>) {
        (get<Is>(operands_tuple).Forward(), ...);
        return (get<Is>(operands_tuple).value() * ...);
    }

    void Forward() {
        whole.whole.data = VarForward(std::index_sequence_for<Operands...>{});
    }

    template<std::size_t... Is>
    void VarBackward(std::index_sequence<Is...>) {
        auto product = (get<Is>(operands_tuple).value() * ...);
        ((get<Is>(operands_tuple).whole.grad = (whole.grad * product / get<Is>(operands_tuple).value())), ...);

        (get<Is>(operands_tuple).Backward(), ...);
    }

    void Backward() {
        VarBackward(std::index_sequence_for<Operands...>{});
    }
};

// 多个变量求和
template<typename... Operands>
struct VarAddition {

    using Whole = Variable<double>;

    std::tuple<operand_ref_t<Operands>...> operands_tuple;

    Whole whole{}; // 看成一个整体

    explicit VarAddition(const Operands &... operands) : operands_tuple(const_cast<Operands &>(operands)...) {}

    template<std::size_t... Is>
    auto VarForward(std::index_sequence<Is...>) {
        (get<Is>(operands_tuple).Forward(), ...);
        return (get<Is>(operands_tuple).whole.data + ...);
    }

    void Forward() {
        whole.data = VarForward(std::index_sequence_for<Operands...>{});
    }

    template<std::size_t... Is>
    void VarBackward(std::index_sequence<Is...>) {
        ((get<Is>(operands_tuple).whole.grad = whole.grad), ...);

        (get<Is>(operands_tuple).Backward(), ...);
    }

    void Backward() {
        VarBackward(std::index_sequence_for<Operands...>{});
    }
};


// 向量看成列数为1的矩阵
template<std::size_t N, typename InDataType = double>
using Vector = Matrix<N, 1, InDataType>;

// 矩阵变量
template<std::size_t InN, std::size_t InM>
struct MatrixVariable {

    using MatrixVariableClassFlag = std::nullptr_t;

    static constexpr std::size_t N = InN;
    static constexpr std::size_t M = InM;
    using Mat = Matrix<N, M>;

    Mat data; // 变量的值
    Mat grad; // 变量的梯度

    MatrixVariable &whole;

    void step(double lr) {
        data += -lr * grad;
    }

    void clear_grad() {
        grad = Mat{};
    }

    void init_grad() {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                grad.data[i][j] = 1;
            }
        }
    }

    MatrixVariable() : whole(*this) { init_grad(); }

    explicit MatrixVariable(const Matrix<InN, InM> &in_data) : data(in_data), whole(*this) { init_grad(); }

    void set_value(const Mat &value) { data = value; }

    void Forward() {}

    void Backward() {}
};

template<typename InOperand1, typename InOperand2>
struct TwoOperandPolicy {
    using Operand1 = InOperand1;
    using Operand2 = InOperand2;
};

template<MatrixLikeComponent InOperand1, MatrixLikeComponent InOperand2>
struct MatMultiplyPolicy : TwoOperandPolicy<InOperand1, InOperand2> {
    static constexpr std::size_t N = InOperand1::N;
    static constexpr std::size_t M = InOperand2::M;

    void Forward(InOperand1& operand1, InOperand2& operand2, auto& whole) {
        operand1.Forward();
        operand2.Forward();
        whole.set_value(operand1.whole.data * operand2.whole.data);
    }

    void Backward(InOperand1& operand1, InOperand2& operand2, auto& whole) {
        // (N, K) * (K, M) => (N, M)
        // (N, M) * (M, K) => (N, K)
        // (K, N) * (N, M) => (K, M)

        operand1.whole.grad = whole.grad * operand2.whole.data.T();
        operand2.whole.grad = operand1.whole.data.T() * whole.grad;

        operand1.Backward();
        operand2.Backward();
    }
};

template<MatrixLikeComponent InOperand1, MatrixLikeComponent InOperand2>
struct MatAdditionPolicy : TwoOperandPolicy<InOperand1, InOperand2> {
    static constexpr std::size_t N = InOperand1::N;
    static constexpr std::size_t M = InOperand1::M;

    void Forward(InOperand1& operand1, InOperand2& operand2, auto& whole) {
        operand1.Forward();
        operand2.Forward();

        whole.data = operand1.whole.data + operand2.whole.data;
    }

    void Backward(InOperand1& operand1, InOperand2& operand2, auto& whole) {
        operand1.whole.grad = whole.grad;
        operand2.whole.grad = whole.grad;

        operand1.Backward();
        operand2.Backward();
    }
};

template<MatrixLikeComponent InOperand1, MatrixLikeComponent InOperand2, template<class, class> class Policy>
struct MatBinaryOperator {

    using Operand1 = InOperand1;
    using Operand2 = InOperand2;

    using Whole = MatrixVariable<InOperand1::N, InOperand2::M>;

    static constexpr std::size_t N = InOperand1::N;
    static constexpr std::size_t M = InOperand2::M;

    Whole whole;

    operand_ref_t<Operand1> operand1; // N * K
    operand_ref_t<Operand2> operand2; // K * M

    Policy<InOperand1, InOperand2> policy;

    MatBinaryOperator(const InOperand1 &o1, const InOperand2 &o2) : operand1(const_cast<InOperand1 &>(o1)),
                                                              operand2(const_cast<InOperand2 &>(o2)) {}

    void Forward() {
        policy.Forward(operand1, operand2, whole);
    }


    void Backward() {
        policy.Backward(operand1, operand2, whole);
    }
};

// 矩阵相乘
template<MatrixLikeComponent InOperand1, MatrixLikeComponent InOperand2>
using MatMultiply = MatBinaryOperator<InOperand1, InOperand2, MatMultiplyPolicy>;

// 矩阵相加
template<MatrixLikeComponent InOperand1, MatrixLikeComponent InOperand2>
using MatAddition = MatBinaryOperator<InOperand1, InOperand2, MatAdditionPolicy>;


// 对矩阵的每一维求指数运算
template<MatrixLikeComponent InOperand>
struct Power {

    using Operand = InOperand;

    SINGLE_OPERAND_MATRIX_NM

    using Whole = MatrixVariable<N, M>;

    const unsigned int K; //  求K次方

    operand_ref_t<Operand> operand;
    Whole whole;

    Power(unsigned int k, const InOperand &o) : K(k), operand(const_cast<InOperand &>(o)) {}

    void Forward() {
        operand.Forward();

        auto &val = operand.whole.data;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                whole.data[i][j] = std::pow(val[i][j], K);
            }
        }
    }

    void Backward() {
        auto &val = operand.whole.data;

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                operand.whole.grad[i][j] = whole.grad[i][j] * K * std::pow(val[i][j], K - 1);
            }
        }

        operand.Backward();
    }
};

// 对矩阵的每个元素进行映射 f(x) = 1 / (1 + exp(-x))
template<MatrixLikeComponent InOperand>
struct Sigmoid {

    using Operand = InOperand;

    SINGLE_OPERAND_MATRIX_NM

    using Whole = MatrixVariable<N, M>;

    operand_ref_t<Operand> operand;
    Whole whole;

    explicit Sigmoid(const InOperand &o) : operand(const_cast<InOperand &>(o)) {}

    Matrix<Operand::N, Operand::M> &value() { return whole.data; }

    void Forward() {
        operand.Forward();

        auto &val = operand.whole.data;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                auto &x = val.data[i][j];
                whole.data[i][j] = 1 / (1 + std::exp(-x));
            }
        }
    }

    void Backward() {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                operand.whole.grad[i][j] = whole.grad[i][j] * (1 - whole.data[i][j]) * whole.data[i][j];
            }
        }

        operand.Backward();
    }
};

// 矩阵乘标量（标量不可优化）
template<MatrixLikeComponent InOperand, typename ScalarType>
struct MatScale {
    SINGLE_OPERAND_MATRIX_NM

    using Operand = InOperand;
    using Whole = MatrixVariable<InOperand::N, InOperand::M>;


    Whole whole;

    operand_ref_t<InOperand> operand;
    ScalarType scalar;

    MatScale(const Operand &o, ScalarType scalar_) : operand(const_cast<InOperand &>(o)), scalar(scalar_) {}

    void Forward() {
        operand.Forward();

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                whole.data[i][j] = scalar * operand.whole.data[i][j];
            }
        }
    }

    void Backward() {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                operand.whole.grad[i][j] = scalar * whole.grad[i][j];
            }
        }

        operand.Backward();
    }
};

// 将矩阵的所有元素加起来
template<MatrixLikeComponent InOperand>
struct Sum {

    using Operand = InOperand;

    using Whole = Variable<double>;

    Whole whole;

    operand_ref_t<Operand> operand;

    explicit Sum(const InOperand &o) : operand(const_cast<InOperand &>(o)) {}

    void Forward() {
        operand.Forward();

        whole.whole.data = 0;

        auto &val = operand.whole.data;
        for (int i = 0; i < InOperand::N; i++) {
            for (int j = 0; j < InOperand::M; j++) {
                whole.whole.data = whole.value() + val[i][j];
            }
        }
    }

    void Backward() {
        for (int i = 0; i < InOperand::N; i++) {
            for (int j = 0; j < InOperand::M; j++) {
                operand.whole.grad[i][j] = whole.gradient();
            }
        }

        operand.Backward();
    }
};

template<std::size_t InFilterSize, std::size_t InStride, MatrixLikeComponent InOperand>
struct Conv2d {

    static constexpr std::size_t N = (InOperand::N - InFilterSize) / InStride + 1;
    static constexpr std::size_t M = (InOperand::M - InFilterSize) / InStride + 1;
    using Whole = MatrixVariable<N, M>;

    MatrixVariable<InFilterSize, InFilterSize> filter;

    Whole whole;

    operand_ref_t<InOperand> operand;

    static constexpr std::size_t FilterSize = InFilterSize;
    static constexpr std::size_t Stride = InStride;

    explicit Conv2d(const InOperand &o) : operand(const_cast<InOperand &>(o)) {}

    Optimizable auto &parameters() { return filter; }

    // 初始化filter
    void initialize(const Matrix<InFilterSize, InFilterSize> &params) { filter.set_value(params); }

    void Forward() {
        operand.Forward();

        for (std::size_t i = 0; i < N; i++) {
            for (std::size_t j = 0; j < M; j++) {
                whole.data[i][j] = 0;
                for (std::size_t u = Stride * i, f_i = 0; f_i < FilterSize; f_i++) {
                    for (std::size_t v = Stride * j, f_j = 0; f_j < FilterSize; f_j++) {
                        whole.data[i][j] += filter.data[f_i][f_j] * operand.whole.data[u + f_i][v + f_j];
                    }
                }
            }
        }
    }

    void Backward() {
        for (std::size_t i = 0; i < N; i++) {
            for (std::size_t j = 0; j < M; j++) {
                for (std::size_t u = Stride * i, f_i = 0; f_i < FilterSize; f_i++) {
                    for (std::size_t v = Stride * j, f_j = 0; f_j < FilterSize; f_j++) {
                        operand.whole.grad[u + f_i][v + f_j] += whole.grad[i][j] * filter.data[f_i][f_j];
                        filter.grad[f_i][f_j] += whole.grad[i][j] * operand.whole.data[u + f_i][v + f_j];
                    }
                }
            }
        }

        operand.Backward();
    }
};

template<std::size_t InDim, std::size_t OutDim, MatrixLikeComponent InOperand>
struct FullyConnected {

    static constexpr std::size_t N = OutDim;
    static constexpr std::size_t M = 1;

    using Whole = MatrixVariable<N, M>;
    Whole whole;

    MatrixVariable<OutDim, InDim> w;
    MatrixVariable<OutDim, 1> b;

    operand_ref_t<InOperand> operand;

    explicit FullyConnected(const InOperand &o) : operand(const_cast<InOperand &>(o)) {}


    void Forward() {
        operand.Forward();

        whole.data = w.data * operand.whole.data + b.data;
    }

    void Backward() {
        w.grad = whole.grad * operand.whole.data.T();
        operand.whole.grad = w.data.T() * whole.grad;
        b.grad = whole.grad;

        operand.Backward();
    }
};

template<Optimizable... T>
inline void step(double lr, T &... variables) {
    (variables.step(lr), ...);
    (variables.clear_grad(), ...);
}

template<typename T>
using pure_t = std::remove_cvref_t<T>;

template<MatrixLikeComponent LHS, MatrixLikeComponent RHS>
inline auto operator*(const LHS &lhs, const RHS &rhs) {
    return MatMultiply<pure_t<decltype(lhs)>, pure_t<decltype(rhs)>>{lhs, rhs};
}

template<Scalar ScalarType, MatrixLikeComponent RHS>
inline auto operator*(ScalarType scalar, const RHS &rhs) {
    return MatScale{rhs, scalar};
}

template<Scalar ScalarType, MatrixLikeComponent LHS>
inline auto operator*(const LHS &lhs, ScalarType scalar) {
    return MatScale{lhs, scalar};
}

template<MatrixLikeComponent LHS, MatrixLikeComponent RHS>
requires (IsMatrixVariable<LHS> || IsMatrixVariable<typename LHS::Whole>) &&
         (IsMatrixVariable<RHS> || IsMatrixVariable<typename RHS::Whole>)
inline auto operator+(const LHS &lhs, const RHS &rhs) {
    return MatAddition<pure_t<decltype(lhs)>, pure_t<decltype(rhs)>>{lhs, rhs};
}

template<Component LHS, Component RHS>
requires (IsSingleVariable<LHS> || IsSingleVariable<typename LHS::Whole>) &&
         (IsSingleVariable<RHS> || IsSingleVariable<typename RHS::Whole>)
inline auto operator+(const LHS &lhs, const RHS &rhs) {
    return VarAddition{lhs, rhs};
}

template<MatrixLikeComponent LHS, MatrixLikeComponent RHS>
inline auto operator-(const LHS &lhs, const RHS &rhs) {
    return lhs + (-rhs);
}

// 用^模拟指数运算符，注意^的优先级低于+
template<MatrixLikeComponent LHS>
inline auto operator^(const LHS &lhs, unsigned int k) {
    return Power{k, lhs};
}

// 乘以-1
template<MatrixLikeComponent LHS>
inline auto operator-(const LHS &lhs) {
    return MatScale{lhs, -1};
}

#endif //METAAI_COMPONENTS_H
