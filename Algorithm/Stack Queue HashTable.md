## Intro
- Array和Linked List都可以实现Stack和Queue
- Array和Linked List是最基本的物理结构 其他数据结构都是依赖于物理结构的逻辑结构
- Stack和Queue都是线性数据结构

## Stack
栈顶端 `top` 加入数据 (`push`) 输出数据 (`pop`)
"后进先出" Last In First Out (LIFO)
- `push` 入栈 `O(1)`
- `pop` 出栈 `O(1)`
- Python里面的list很好地实现了stack的功能，`append`相当于入栈 `pop`相当于出栈
- 递归的逻辑用stack实现 因为涉及到对“历史”的回溯

## Queue
"先进先出" First In First Out (FIFO)。 队头 `front` 队尾 `rear`
- `enqueue` 入队 `O(1)`
- `dequeue` 出队 `O(1)`

## Deque 双端队列
- 综合了stack和queue的优点，队头可以入队和出队，队尾也可以入队和出队，相当于两端都是一个stack

## Hash Table 哈希表
- 本质上是一个Array 存储`Key-Value`映射的集合。对于某一个`Key`哈希表可以在接近`O(1)`的时间进行读写操作
- 因为是Array当我们读取的时候首先把`Key`利用hash function 转换为index。在Python里面每一个object都有一个独一无二的hash value 无论变量的类型是什么 hash value都是一个int
- **Hash Collision** 哈希冲突是说两个不同的value算出来的hash value也就是index是相同的 解决办法是：
	- **开放寻址** 按照某一个pattern找剩下的index是否有空位直到找到为止。Python里的`dict`就是用的这个方法。
	- **链表法**  就是把hash table里面的每个entry存成一个链表，而且是头节点，每个entry对象通过next指针指向它的下一个entry节点。当新来的entry映射到与之冲突的数组位置时，只需插入对应的链表中即可，也就是把原来与之冲突的entry的next指针指向新来的entry
