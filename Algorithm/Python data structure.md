## Pandas.DataFrame
> **Data alignment is intrinsic**

### Series
Series is a one-dimensional labeled array capable of holding any data type (integers, strings, floating point numbers, Python objects, etc.). The axis labels are collectively referred to as the **index**.

A key difference between Series and **ndarray** is that operations between Series automatically align the data based on label. Thus, you can write computations without giving consideration to whether the Series involved have the same labels.
e.g.
```python
s = pd.Series({'b': 1, 'a': 11, 'c': 10})
s[1:] + s[:-1]
# a    22.0
# b     NaN
# c     NaN
```

### DataFrame
**DataFrame** is a 2-dimensional labeled data structure with columns of potentially different types. You can think of it like a spreadsheet or SQL table, or a dict of Series objects. Along with the data, you can optionally pass **index** (row labels) and **columns** (column labels) arguments.
```python
import pandas as pd

d = {'one': [1,2,3,4], 'two': [4, 3, 2, 1]}
df = pd.DataFrame(d, index = ['a', 'b', 'c', 'd'])

#    one  two
# a  1.0  4.0
# b  2.0  3.0
# c  3.0  2.0
# d  4.0  1.0

data2 = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]

pd.DataFrame(data2, index=['first', 'second'])
#               a   b     c
# first   	1   2   NaN
# second  	5  10  20.0

pd.DataFrame(data2, columns=['a', 'b'])
#    a   b
# 0  1   2
# 1  5  10
```


## Numpy arrays
A numpy array is a grid of values, all of the same type, and is indexed by a tuple of nonnegative integers.
```python
import numpy as np
a = np.array([1,2,3])
print(a.shape)
# (3,)

b = np.array([[1,2,3], [4,5,6]])
print(b.shape)
# (2,3)

a = np.zeros((2,2))
b = np.ones((1,2))
c = np.full((2,2), 7) # create a constant array filling in 7 with shape (2,2)
d = np.random.random((2,2)) # create a random array with random values with shape (2,2)
```
`*` is element-wise multiplication, not matrix multiplication. We instead use the `dot` function to compute inner products of vectors, to multiply a vector by a matrix, and to multiply matrices.

```python
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
				
y = np.empty_like(x)   # Create an empty matrix with the same shape as x
z = np.empty(shape=(4,3)) # Another way
```
Difference between `(R,)` and `(1, R)`
```python
a = np.arange(12)
# array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
b = a.reshape((1,12)
b.shape
# (1,12)
b
# array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

```

## Stack implementation in Python
```python
from collections import deque
s = deque()  
```
### Differences between `list` and `deque`  
- `list`: faster in accessing any random element;  block memory in the sense that items are stored next to each other in memory  
- `deque`: "doubly linked list", each entry has references to both the previous and the next entry in the list; constant time in `.append()` and `.pop()` operations but slow in getting any item

## Heap implementation in Python
```python
import heapq
# init
heap_lst = []
for item in nums:  
  heapq.heappush(heap_lst, item)  
print(heap_lst)
  
# another way
heapq.heapify(nums) # modifies the list in place but does not sorting it

heapq.heappop(nums) # modifies the list by removing the first item  
heapq.heappush(nums, 4) #pushing an element to the heap while preserving the heap property
```
`heapreplace()` is equivalent to `heappop()` followed by `heappush()`.  
`heappushpop()` is equivalent to `heappush()` followed by `heappop()`.

### Where to use it
A `heap` is an implementation of complete binary tree. Useful for finding the K largest (**min heap**) or smallest number (**max heap**) in a long array or list.

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEwNjQ0MzU0OTUsMjUxNTc1MTQ4LC0yMD
cwMjcyODE1LDIzOTc5MDQ2NSw3MTU1MTIzNjMsNDUwODQwODU0
LDE5MTk5OTYxMzgsLTQ3MTU4MzA3NywxMTA5Mzc2ODQ5LDYwMz
g5NTc3NV19
-->