import numpy as np

def solution(a):
    cnt = 0
    i = 0
    while i < len(a)-1:
        if np.max(a[:i+1]) < np.min(a[i+1:]):
            cnt += 1
        i +=1
    return cnt+1

if __name__ == '__main__':
    test_case_1 = [2,4,1,6,5,9,7]
    test_case_2 = [2,1,6,4,3,7]
    assert solution(test_case_1) == 3
    assert solution(test_case_2) == 3