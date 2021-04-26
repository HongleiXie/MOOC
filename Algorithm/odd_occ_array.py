#找数组里面出现odd次数的数 其余出现even次数
from collections import Counter

def no_use_counter(a):
    # d = dict(Counter(a))
    d = dict((x, a.count(x)) for x in set(a))
    # return list(dict(filter(lambda elem: elem[1] % 2 == 1, d.items())))
    return [key for (key, value) in d.items() if value %2 == 1]


# no_use_counter([9,3,10,10,10,9,8,8])
# Out: [10, 3]

# no_use_counter([])
# []

# no_use_counter([1,1])
# []