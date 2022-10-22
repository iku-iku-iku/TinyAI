C++元编程实现AI框架

> 元编程真好玩，就是头有点凉

编译器需要支持C++20

实现了若干Operator，目前利用Operator来组装网络

重载了+*^-操作符方便模型构建

主要思路为：
1. Forward时把计算结果存储到whole中 (后序遍历：先对Operand进行Forward，在计算当前Component的结果)
2. Backward时利用whole的grad来计算operand的grad (前序遍历：先计算operand的grad，再对operand进行Backward)

使用到的新特性
1. auto返回值类型推导
2. 类模板参数推导
3. concept和requires
4. fold expression
5. alias template