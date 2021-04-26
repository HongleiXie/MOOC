# find the split between winter and summer such that
# each temperature in the winter should be smaller than the measurement of summer's temperature
# make the winter as short as possible
# return the length of the winter
import numpy as np

def solution(a):
    i = 0
    while i < len(a)-1:
        if np.max(a[:i+1]) < np.min(a[i+1:]):
            return i+1
        i +=1

if '__name__' == '__main__':
    test_case_1 = [5, -2, 3, 8, 6]
    test_case_2 = [-5, -5, -5, -42, 6, 12]
    assert solution(test_case_1) == 3
    assert solution(test_case_2) == 4