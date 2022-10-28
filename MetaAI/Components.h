//
// Created by iku-iku-iku on 2022/10/21.
//

#ifndef METAAI_COMPONENTS_H
#define METAAI_COMPONENTS_H

#include <functional>
#include "policies.h"
#include <iostream>
#include "Type.h"
#include <tuple>
#include <cstring>
#include <cmath>

template<typename InDataType>
struct SingleVariable;

template<std::size_t InN, std::size_t InM>
struct MatrixVariable;

template<typename OperandType>
struct z_operand_ref {
    using type = std::remove_reference_t<OperandType>;
};

// 目前是完全在栈上运算，效率更高，因此需要对于栈上的Variable需要引用
template<typename InDataType>
struct z_operand_ref<SingleVariable<InDataType>> {
    using type = std::remove_reference_t<SingleVariable<InDataType>> &;
};

template<std::size_t InN, std::size_t InM>
struct z_operand_ref<MatrixVariable<InN, InM>> {
    using type = std::remove_reference_t<MatrixVariable<InN, InM>> &;
};

template<typename InOperandType>
using operand_ref_t = typename z_operand_ref<InOperandType>::type;

template<typename T>
struct z_data_type_of {
};

template<typename T>
using data_type_of_t = typename z_data_type_of<T>::type;

struct Variable {
    virtual void step(double lr) = 0;

    virtual void clear_grad() = 0;
};

// 单独的变量
template<typename InDataType>
struct SingleVariable : Variable {

    using VariableClassFlag = std::nullptr_t;

    using DataType = InDataType;

    DataType data{};
    DataType grad{1}; // 对自己的梯度为1

    SingleVariable &whole;

    SingleVariable() : whole(*this) {}

    explicit SingleVariable(DataType x) : data(x), whole(*this) {}

    friend auto &operator<<(std::ostream &out, const SingleVariable &var) {
        return out << var.data;
    }

    DataType value() { return data; }

    DataType gradient() { return grad; }

//    Variable(const Variable &) = delete;
//
//    Variable &operator=(const Variable &) = delete;

    // 利用梯度更新参数
    void step(double lr) override {
        data += -lr * grad;
    }

    // 清除梯度
    void clear_grad() override {
        grad = DataType{};
    }

    void Forward() {}

    void Backward() {}
};

template<typename T>
struct z_data_type_of<SingleVariable<T>> {
    using type = typename SingleVariable<T>::DataType;
};

template<typename... Operands>
struct VarMultiply {

    using Whole = SingleVariable<double>;

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

    using Whole = SingleVariable<double>;

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
struct MatrixVariable : Variable {

    using MatrixVariableClassFlag = std::nullptr_t;

    static constexpr std::size_t N = InN;
    static constexpr std::size_t M = InM;
    using Mat = Matrix<N, M>;

    Mat data; // 变量的值
    Mat grad; // 变量的梯度

    MatrixVariable &whole;

    void step(double lr) override {
        data += -lr * grad;
    }

    void clear_grad() override {
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

template<std::size_t N, std::size_t M>
struct z_data_type_of<MatrixVariable<N, M>> {
    using type = typename MatrixVariable<N, M>::Mat;
};

template<C_MatrixLikeComponent InOperand1, C_MatrixLikeComponent InOperand2, template<class, class> class Policy>
struct MatBinaryOperator {

    using Operand1 = InOperand1;
    using Operand2 = InOperand2;

    using Whole = MatrixVariable<InOperand1::N, InOperand2::M>;
    using PolicyType = Policy<InOperand1, InOperand2>;

    static constexpr std::size_t N = PolicyType::N;
    static constexpr std::size_t M = PolicyType::M;

    Whole whole;

    operand_ref_t<Operand1> operand1; // N * K
    operand_ref_t<Operand2> operand2; // K * M

    PolicyType policy;

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
template<C_MatrixLikeComponent InOperand1, C_MatrixLikeComponent InOperand2>
using MatMultiply = MatBinaryOperator<InOperand1, InOperand2, MatMultiplyPolicy>;

// 矩阵相加
template<C_MatrixLikeComponent InOperand1, C_MatrixLikeComponent InOperand2>
using MatAddition = MatBinaryOperator<InOperand1, InOperand2, MatAdditionPolicy>;

template<C_MatrixLikeComponent InOperand, template<class> class InPolicy>
struct ElementwiseOperator {

    using Operand = InOperand;

    static constexpr std::size_t N = InOperand::N;
    static constexpr std::size_t M = InOperand::M;

    using Whole = MatrixVariable<N, M>;
    using Policy = InPolicy<InOperand>;

    operand_ref_t<Operand> operand;
    Whole whole;
    Policy policy;

    explicit ElementwiseOperator(const InOperand &o) : operand(const_cast<InOperand &>(o)) {}

    void Forward() {
        policy.Forward(operand, whole);
    }

    void Backward() {
        policy.Backward(operand, whole);
    }
};

// 对矩阵的每个元素进行映射 f(x) = x ^ k
template<C_MatrixLikeComponent InOperand>
using Power = ElementwiseOperator<InOperand, PowerPolicy>;

// 对矩阵的每个元素进行映射 f(x) = 1 / (1 + exp(-x))
template<C_MatrixLikeComponent InOperand>
using Sigmoid = ElementwiseOperator<InOperand, SigmoidPolicy>;

// 矩阵乘标量（标量不可优化）
template<C_MatrixLikeComponent InOperand>
using MatScale = ElementwiseOperator<InOperand, ScalePolicy>;

// 将矩阵的所有元素加起来
template<C_MatrixLikeComponent InOperand>
struct Sum {

    using Operand = InOperand;

    using Whole = SingleVariable<double>;

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

// 将一系列层组成一个复合函数，整体可以看成一个Component，能够与其他Component进行组合
template<std::size_t InN, std::size_t InM, typename... Layer>
struct Sequence {

    std::tuple<Layer...> layers{};

    static constexpr std::size_t N = InN;
    static constexpr std::size_t M = InM;

    // 使得loss可以通过sequence进行Backward
    struct Whole : MatrixVariable<N, M>{
        Sequence& seq;
        Whole(Sequence& seq_) : seq(seq_) {}

        void Backward() {
            seq.Backward();
        }
    };

    Whole whole;

    Sequence() : whole(*this) {}

    // 将sequence看作复合函数，并且返回封装后的whole，可以看作一个component，参与构建
    auto &operator()(const auto &input) {
        Forward(input);
        return whole;
    }

    // 将input逐层传递
    void Forward(const auto &input) {
        whole.data = ForwardHelper<0, sizeof...(Layer)>::Forward(input, layers);
    }

    template<std::size_t I, std::size_t MAX>
    struct ForwardHelper {
        static auto Forward(const auto &input, auto &layers) {
            if constexpr (I == MAX) { return input; }
            else { return ForwardHelper<I + 1, MAX>::Forward(get<I>(layers).Forward(input), layers); }
        }
    };

    // 将梯度逐层反向传回
    void Backward() {
        BackwardHelper<sizeof...(Layer) - 1>::Backward(whole.grad, layers);
    }

    template<std::size_t I>
    struct BackwardHelper {
        static auto Backward(const auto &grad, auto &layers) {
            if constexpr (I == 0) { return get<I>(layers).Backward(grad); }
            else { return BackwardHelper<I - 1>::Backward(get<I>(layers).Backward(grad), layers); }
        }
    };

    template<typename... Elem1, typename... Elem2>
    inline auto compose(const std::tuple<Elem1 &...> &t1, const std::tuple<Elem2 &...> &t2) {
        return std::tuple_cat(t1, t2);
    }

    template<std::size_t I, std::size_t MAX, typename... Elem>
    auto parameters(const std::tuple<Elem &...> &t) {
        if constexpr (I == MAX) { return t; }
        else if constexpr (C_ParameterLayer<decltype(get<I>(layers))>) {
            return parameters<I + 1, MAX>(compose(t, get<I>(layers).parameters()));
        } else {
            return parameters<I + 1, MAX>(t);
        }
    }

    auto parameters() {
        return parameters<0, sizeof...(Layer)>(std::make_tuple());
    }
};

struct TrivialPassLayer {
    auto Forward(const auto &input) {
        return input;
    }

    auto Backward(const auto &grad) {
        return grad;
    }
};

template<std::size_t N>
struct ScaleLayer {
    auto Forward(const auto &input) {
        return input * N;
    }

    auto Backward(const auto &grad) {
        return N * grad;
    }
};

template<std::size_t N, std::size_t M>
struct SumLayer {
    auto Forward(const Matrix<N, M> &input) {
        typename Matrix<N, M>::DataType out{};

        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < M; ++j) {
                out += input.get(i, j);
            }
        }

        return out;
    }

    auto Backward(const typename Matrix<N, M>::DataType &grad) {
        Matrix<N, M> res;

        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < M; ++j) {
                res.get(i, j) = grad;
            }
        }

        return res;
    }
};

template<std::size_t N, std::size_t M, std::size_t K>
struct PowerLayer {
    Matrix<N, M> cached_input;

    auto Forward(const Matrix<N, M> &input) {
        cached_input = input;
        Matrix<N, M> out;

        auto &val = input.data;
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < M; ++j) {
                out.get(i, j) = std::pow(input.get(i, j), K);
            }
        }

        return out;
    }

    auto Backward(const Matrix<N, M> &grad) {
        Matrix<N, M> res;

        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < M; ++j) {
                res.get(i, j) = grad.get(i, j) * K * std::pow(cached_input.get(i, j), K - 1);
            }
        }

        return res;
    }
};

template<std::size_t N, std::size_t M>
struct SigmoidLayer {
    Matrix<N, M> cached_out;

    auto Forward(const Matrix<N, M> &input) {
        Matrix<N, M> out;
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < M; ++j) {
                auto &x = input.get(i, j);
                out.get(i, j) = 1 / (1 + std::exp(-x));
            }
        }
        cached_out = out;
        return out;
    }

    auto Backward(const Matrix<N, M> &grad) {
        Matrix<N, M> res;
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < M; ++j) {
                res.get(i, j) = grad.get(i, j) * (1 - cached_out.get(i, j)) * cached_out.get(i, j);
            }
        }
        return res;
    }
};

template<std::size_t InDim, std::size_t OutDim, typename Initializer>
struct FullyConnectedLayer {
    static constexpr std::size_t N = OutDim;
    static constexpr std::size_t M = 1;

    MatrixVariable<OutDim, InDim> w;
    MatrixVariable<OutDim, 1> b;

    using Input = Matrix<InDim, 1>;
    Input cached_input;

    FullyConnectedLayer() : w(Matrix<OutDim, InDim>::template CreateWith<Initializer>()),
                            b(Matrix<OutDim, 1>::template CreateWith<Initializer>()) {}

    auto Forward(const Input &input) {
        cached_input = input;
        return w.data * input + b.data;
    }

    auto Backward(const auto &grad) {
        w.grad = grad.matmul(cached_input.T());
        b.grad = grad;
        return w.data.T() * grad;
    }

    auto parameters() {
        return std::make_tuple(std::ref(w), std::ref(b));
    }
};

template<std::size_t InN, std::size_t InM, std::size_t InFilterSize, std::size_t InStride, C_MatrixLikeComponent InOperand>
struct Conv2dLayer {
    static constexpr std::size_t N = (InOperand::N - InFilterSize) / InStride + 1;
    static constexpr std::size_t M = (InOperand::M - InFilterSize) / InStride + 1;

    MatrixVariable<InFilterSize, InFilterSize> filter;

    using Input = Matrix<InN, InM>;
    Input cached_input;

    static constexpr std::size_t FilterSize = InFilterSize;
    static constexpr std::size_t Stride = InStride;

    auto &parameters() { return std::make_tuple(std::ref(filter)); }

    auto Forward(const Input& input) {
        cached_input = input;
        Matrix<N, M> out;

        for (std::size_t i = 0; i < N; i++) {
            for (std::size_t j = 0; j < M; j++) {
                for (std::size_t u = Stride * i, f_i = 0; f_i < FilterSize; f_i++) {
                    for (std::size_t v = Stride * j, f_j = 0; f_j < FilterSize; f_j++) {
                        out.get(i, j) += filter.get(f_i, f_j) * input.get(u + f_i, v + f_j);
                    }
                }
            }
        }

        return out;
    }

    auto Backward(const auto& grad) {
        Matrix<InN, InM> res;

        for (std::size_t i = 0; i < N; i++) {
            for (std::size_t j = 0; j < M; j++) {
                for (std::size_t u = Stride * i, f_i = 0; f_i < FilterSize; f_i++) {
                    for (std::size_t v = Stride * j, f_j = 0; f_j < FilterSize; f_j++) {
                        res.get(u + f_i, v + f_j) += grad.get(i, j) * filter.data.get(f_i, f_j);
                        filter.grad.get(f_i, f_j) += grad.get(i, j) * cached_input.get(u + f_i, v + f_j);
                    }
                }
            }
        }

        return res;
    }
};

template<C_Optimizable... T>
inline void step(double lr, T &... variables) {
    (variables.step(lr), ...);
    (variables.clear_grad(), ...);
}

template<typename T>
using pure_t = std::remove_cvref_t<T>;

template<C_MatrixLikeComponent LHS, C_MatrixLikeComponent RHS>
inline auto operator*(const LHS &lhs, const RHS &rhs) {
    return MatMultiply<pure_t<decltype(lhs)>, pure_t<decltype(rhs)>>{lhs, rhs};
}

template<C_Scalar ScalarType, C_MatrixLikeComponent RHS>
inline auto operator*(ScalarType scalar, const RHS &rhs) {
    auto op = MatScale<pure_t<pure_t<decltype(rhs)>>>{rhs};
    op.policy.scalar = scalar;
    return op;
}

template<C_Scalar ScalarType, C_MatrixLikeComponent LHS>
inline auto operator*(const LHS &lhs, ScalarType scalar) {
    auto op = MatScale<pure_t<pure_t<decltype(lhs)>>>{lhs};
    op.policy.scalar = scalar;
    return op;
}

// 乘以-1
template<C_MatrixLikeComponent LHS>
inline auto operator-(const LHS &lhs) {
    return lhs * -1;
}

template<C_MatrixLikeComponent LHS, C_MatrixLikeComponent RHS>
requires (C_MatrixVariable<LHS> || C_MatrixVariable<typename LHS::Whole>) &&
         (C_MatrixVariable<RHS> || C_MatrixVariable<typename RHS::Whole>)
inline auto operator+(const LHS &lhs, const RHS &rhs) {
    return MatAddition<pure_t<decltype(lhs)>, pure_t<decltype(rhs)>>{lhs, rhs};
}

template<C_Component LHS, C_Component RHS>
requires (C_SingleVariable<LHS> || C_SingleVariable<typename LHS::Whole>) &&
         (C_SingleVariable<RHS> || C_SingleVariable<typename RHS::Whole>)
inline auto operator+(const LHS &lhs, const RHS &rhs) {
    return VarAddition{lhs, rhs};
}

template<C_MatrixLikeComponent LHS, C_MatrixLikeComponent RHS>
inline auto operator-(const LHS &lhs, const RHS &rhs) {
    return lhs + (-rhs);
}

// 用^模拟指数运算符，注意^的优先级低于+
template<C_MatrixLikeComponent LHS>
inline auto operator^(const LHS &lhs, unsigned int k) {
    auto op = Power<pure_t<decltype(lhs)>>{lhs};
    op.policy.K = k;

    return op;
}


#endif //METAAI_COMPONENTS_H
