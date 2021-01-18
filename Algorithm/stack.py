# coding: utf-8

class Stack(object):
    def __init__(self):
        self.__list = [] # make it private 容器选择用顺序表

    def push(self, item):
        """
        添加一个新的元素item到栈顶
        """
        self.__list.append(item) #对于顺序表存的话用尾部添加append()是O(1) 用头部存insert()是O(n)更慢

    def pop(self):
        """
        弹出栈顶元素
        """
        return self.__list.pop()

    def peek(self):
        """
        返回栈顶元素
        """
        if self.__list:
            return self.__list[-1] #返回最后一个元素因为我们决定采用尾部添加
        else:
            return None #空列表

    def is_empty(self):
        return self.__list == [] # "" 0 {} () [] are all FALSE

    def size(self):
        return len(self.__list)

if __name__ == "__main__":
    s = Stack()
    s.push(1)
    s.push(2)
    s.push(3)
    print(s.pop())
    print(s.pop())
    print(s.pop())
