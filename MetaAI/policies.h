//
// Created by iku-iku-iku on 2022/10/27.
//

#ifndef METAAI_POLICY_H
#define METAAI_POLICY_H

#include <cmath>
#include "concepts.h"

template<typename InOperand1, typename InOperand2>
struct TwoOperandPolicy {
    using Operand1 = InOperand1;
    using Operand2 = InOperand2;
};

template<typename InOperand>
struct ElementwisePolicy {
    using Operand = InOperand;
};

template<C_MatrixLikeComponent InOperand>
struct PowerPolicy : ElementwisePolicy<InOperand> {
    std::size_t K;

    void Forward(InOperand &operand, auto &whole) {
        operand.Forward();

        auto &val = operand.whole.data;
        for (int i = 0; i < InOperand::N; i++) {
            for (int j = 0; j < InOperand::M; j++) {
                whole.data[i][j] = std::pow(val[i][j], K);
            }
        }
    }

    void Backward(InOperand &operand, auto &whole) {
        auto &val = operand.whole.data;

        for (int i = 0; i < InOperand::N; i++) {
            for (int j = 0; j < InOperand::M; j++) {
                operand.whole.grad.get(i, j) = whole.grad.get(i, j) * K * std::pow(val.get(i, j), K - 1);
            }
        }

        operand.Backward();
    }
};

template<C_MatrixLikeComponent InOperand>
struct SigmoidPolicy : ElementwisePolicy<InOperand> {
    void Forward(InOperand &operand, auto &whole) {
        operand.Forward();

        auto &val = operand.whole.data;
        for (int i = 0; i < InOperand::N; i++) {
            for (int j = 0; j < InOperand::M; j++) {
                auto &x = val.data[i][j];
                whole.data[i][j] = 1 / (1 + std::exp(-x));
            }
        }
    }

    void Backward(InOperand &operand, auto &whole) {
        for (int i = 0; i < InOperand::N; i++) {
            for (int j = 0; j < InOperand::M; j++) {
                operand.whole.grad[i][j] = whole.grad[i][j] * (1 - whole.data[i][j]) * whole.data[i][j];
            }
        }

        operand.Backward();
    }
};

template<C_MatrixLikeComponent InOperand>
struct ScalePolicy : ElementwisePolicy<InOperand> {
    double scalar;

    void Forward(InOperand& operand, auto& whole) {
        operand.Forward();

        for (int i = 0; i < InOperand::N; ++i) {
            for (int j = 0; j < InOperand::M; ++j) {
                whole.data[i][j] = scalar * operand.whole.data[i][j];
            }
        }
    }

    void Backward(InOperand& operand, auto& whole) {
        for (int i = 0; i < InOperand::N; ++i) {
            for (int j = 0; j < InOperand::M; ++j) {
                operand.whole.grad[i][j] = scalar * whole.grad[i][j];
            }
        }

        operand.Backward();
    }
};

template<C_MatrixLikeComponent InOperand1, C_MatrixLikeComponent InOperand2>
struct MatMultiplyPolicy : TwoOperandPolicy<InOperand1, InOperand2> {
    static constexpr std::size_t N = InOperand1::N;
    static constexpr std::size_t M = InOperand2::M;

    void Forward(InOperand1 &operand1, InOperand2 &operand2, auto &whole) {
        operand1.Forward();
        operand2.Forward();
        whole.set_value(operand1.whole.data * operand2.whole.data);
    }

    void Backward(InOperand1 &operand1, InOperand2 &operand2, auto &whole) {
        // (N, K) * (K, M) => (N, M)
        // (N, M) * (M, K) => (N, K)
        // (K, N) * (N, M) => (K, M)

        operand1.whole.grad = whole.grad * operand2.whole.data.T();
        operand2.whole.grad = operand1.whole.data.T() * whole.grad;

        operand1.Backward();
        operand2.Backward();
    }
};

template<C_MatrixLikeComponent InOperand1, C_MatrixLikeComponent InOperand2>
struct MatAdditionPolicy : TwoOperandPolicy<InOperand1, InOperand2> {
    static constexpr std::size_t N = InOperand1::N;
    static constexpr std::size_t M = InOperand1::M;

    void Forward(InOperand1 &operand1, InOperand2 &operand2, auto &whole) {
        operand1.Forward();
        operand2.Forward();

        whole.data = operand1.whole.data + operand2.whole.data;
    }

    void Backward(InOperand1 &operand1, InOperand2 &operand2, auto &whole) {
        operand1.whole.grad = whole.grad;
        operand2.whole.grad = whole.grad;

        operand1.Backward();
        operand2.Backward();
    }
};

#endif //METAAI_POLICY_H
