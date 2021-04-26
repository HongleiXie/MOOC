# coding:utf-8

def brush_count(a):
    brushCount = 0
    prevHeight = 0
    i = 0
    while i < len(a):
        if a[i] > prevHeight:
            brushCount = brushCount + (a[i] - prevHeight)
        prevHeight = a[i]
        i += 1
    return brushCount


if __name__ == '__main__':
    assert brush_count([1, 2, 4, 2, 5]) == 7
    assert brush_count([1, 4, 3, 2, 3, 1]) == 5
    assert brush_count([4, 1, 2, 1, 2, 2]) == 6
