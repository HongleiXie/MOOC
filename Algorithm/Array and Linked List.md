## Array 数组
- 有限个**相同类型**的变量组成的**有序**集合
- 顺序存储 不能打乱存储顺序也不能跳过 定义的时候以及要给出capacity 然后会根据这个capacity分配一段连续完整的内存空间
- 非常高效的随机访问 给定任意的一个index找对应的元素
- 劣势在插入和删除操作
- 适合读操作多、写操作少的场景

### 查找 `O(1)`
### 更新 `O(1)`
### 插入 综合起来 `O(n)`
- 尾部插入 最简单 `my_list.append(6)` `O(1)`
- 中间插：由于数组的每一个元素都有其固定下标所以不得不把插入位置及后面的元素向后移动腾出地方 `my_list.insert(index, value)` `O(n)`
- 数组扩容：超过capacity的时候会创建一个新的array 长度为原来的两倍再把旧数组复制过去  `O(n)`
### 删除 `O(n)`
被删除前面的元素不变 后面的依次往前挪index

## Linked List 链表
- **单链表 Singly Linked List**：非连续的数据结构 由若干节点node组成 每个node由两部分组成：data和指向下一个元素的指针 next 我们只能通过next指针来找到下一个节点
- **双链表 Doubly Linked List**：每个节点除了next还有指向前一个节点的`prev`指针，这样每个节点都能回溯到`head`头节点
- 随机存储，见缝插针地找内存空间 更加适合灵活 频繁插入 删除的场景

### 查找 `O(n)`
### 更新 `O(1)`
### 插入 `O(1)`
### 删除 `O(1)`
## Python unique Data Structure
Python里面并没有array这个概念，而是使用`list`和`tuple`，本质上都是对数组的封装。其中`list`是一个dynamic array 可以随意添加、删除、修改。`Tuple`是不可修改的。
```
from timeit import Timer
# py2: range returns a list; xrange returns an iterable object
# py3: range returns an iterable object

def t1():
	li = []
	for i in range(10000):
		li.append()

def t2():
	li = []
	for i in range(10000)"
		li += [i] # much slower: li = li + [i]

def t3():
	li = [i for i in range(10000)]

def t4():
	li = list(range(10000))

def t5():
	li = []
	for i in range(10000):
		li.extend([i])

def t7():
	li = []
	for i in range(10000):
		li.insert(0,i)

timer1 = Timer("t1()", "from __main__ import t1")
# running the test in a different script so we need to set up the env
# __main__ refers to the current script
print(timer1.timeit(1000))
timer2 = Timer("t2()", "from __main__ import t2")
timer3 = Timer("t3()", "from __main__ import t3")
timer4 = Timer("t4()", "from __main__ import t4")
```
**List implemented in Python**
- 最快的是`t3`和`t4`; `t2 ~ t5`
- `index` `O(1)`
- `append` `O(1)`
- `contains` `O(n)`
- `pop` `O(n)`
- `insert` `O(n)`

**Dict implemented in Python**
- `contains, copy, get item, set item, delete item` `O(1)`
- `iteration` `O(n)`
