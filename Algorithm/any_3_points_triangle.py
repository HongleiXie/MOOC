def area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1)
                + x3 * (y1 - y2)) / 2.0)

def ifInside(x1,y1,x2,y2,x3,y3,x,y):
    a1 =  area(x1,y1,x2,y2,x,y)
    a2 = area(x1, y1, x3, y3, x, y)
    a3 = area(x2,y2,x3,y3,x,y)
    total_a = area(x1,y1,x2,y2,x3,y3)
    if total_a == a1 + a2 + a3:
        return True
    else:
        return False

ifInside(0,0,1,0,0,1,0.5,0.5)



