import numpy as np

def solution(a):
    if np.max(a) <=0:
        return 1
    else:
        temp_list = [x for x in range(1, np.max(a)) if x not in set(a)]
        if temp_list != []:
            return min(temp_list)
        else:
            return np.max(a) +1

if __name__ == '__main__':
    test_case = [0,23]
    assert solution(test_case) == 1