import re
import random
import string

def solution(S):
    all_upper = set([c for c in S if c.isupper()])
    all_lower = set([c.upper() for c in S if c.islower()])
    both_appear = list(all_upper & all_lower)
    if both_appear != []:
        return max(both_appear)
    else:
        return "NO"

def solution2(S):
    all_upper = re.findall("[A-Z]", S)
    all_lower = re.findall("[a-z]", S)
    all_lower = [x.upper() for x in all_lower]
    both_appear = list(set(all_upper) & set(all_lower))
    if both_appear != []:
        return max(both_appear)
    else:
        return "NO"

# test
N = 100
S = ''.join(random.choices(string.ascii_uppercase, k=N)).join(random.choices(string.ascii_lowercase, k=N))
# %timeit solution(S)
# %timeit solution2(S)