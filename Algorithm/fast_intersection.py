lst1 = [15, 9, 10, 56, 23, 78, 5, 4, 9]
lst2 = [9, 4, 5, 36, 47, 26, 10, 45, 87]

a = [[125, 1], [193, 1], [288, 23]]
b = [[108, 1], [288, 1], [193, 11]]

def intersection_set(lst1, lst2):
    return list(set(lst1) & set(lst2))

def intersection_array(a, b):
    a = dict(a)
    b = dict(b)
    result = []
    for k in a:
        if k in b:
            result.append([k])
    return result
