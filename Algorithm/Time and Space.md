# 时间复杂度和空间复杂度

### 大O记法
对于单调的整数函数 `f`，如果存在一个整数函数`g`和实常数`c>0`，使得对于充分大的`n`总有`f(n)<=c\g(n)`，就说函数`g`是`f`的一个渐近函数（忽略常数），记为`f(n)=O(g(n))`。也就是说，在趋向无穷的极限意义下，函数`f`的增长速度受到函数`g`的约束，即`f(n)`是`g(n)`的同数量级函数。
也就是实现全部基本操作的时间。一般指的是**最坏**时间复杂度。

- 常数记为`O(1)`
- 只保留最高阶
- 如果最高阶存在，去掉前面的系数

### 基本规则
- 顺序结构：加法
- 循环：乘法
- 分支：取最大值

`O(1)<O(logn)<O(n)<O(nlogn)<O(n^2)<O(n^3)<O(2^n)<O(n!)<O(n^n)`
