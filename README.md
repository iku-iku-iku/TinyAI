C++元编程实现一个非常简单的AI框架

> 元编程真好玩，就是头有点凉

编译器需要支持C++20

## 架构
### Component
- 由Component组成有向无环图（DAG）

- 模型由Component构建而成

#### operator
- 矩阵相加、相乘等，是比较简单的算子，比如可以用来构建loss

- 使用policy实现策略模式，解耦结构和具体操作

#### sequence
由一系列layer组成，整个sequence可以看成一个Component

sequence重载了调用运算符

sequence能够取得layer的参数，以供训练使用

### Variable
- 分为SingleVariable和MatrixVariable

- 前者主要基于double，后者基于自己实现的Matrix

- Variable具有data和grad两个关键的属性
  - data：Variable中保存的值
  - grad：某一目标函数对该变量的梯度

### Optimizer
根据参数来构建，能够对参数进行step和clear_grad



## 主要思路
1. Forward时把计算结果存储到whole中 (后序遍历：先对Operand进行Forward，再计算当前Component的结果)
2. Backward时利用whole的grad来计算operand的grad (前序遍历：先计算operand的grad，再对operand进行Backward)

## 使用到的新特性
1. auto返回值类型推导
2. 类模板参数推导
3. concept和requires
4. fold expression
5. alias template
