# coding: utf-8

class Queue(object):
    def __init__(self):
        self.__list = [] #空列表保存数据 顺序表

    def enqueue(self, item):
        """
        增加一个元素
        """
        self.__list.append(item) #增加这个操作比删除更加频繁 O(1)
        # self.__list.insert(0, item) #如果删除这个操作比增加更加频繁 O(n)

    def dequeue(self):
        return self.__list.pop(0) #增加这个操作比删除更加频繁 O(n)
        # return self.__list.pop() #如果删除这个操作比增加更加频繁 O(1)

    def is_empty(self):
        return self.__list == []

    def size(self):
        return len(self.__list)

class Deque(object):
    def __init__(self):
        self.__list = []

    def add_front(self):
        self.__list.insert(0, item)

    def add_rear(self, item):
        self.__list.append(item)

    def pop_front(self):
        return self.__list.pop(0)

    def pop_rear(self):
        return self.__list.pop()


if __name__ == "__main__":
    s = Queue()
    s.enqueue(1)
    s.enqueue(2)
    s.enqueue(3)
    s.enqueue(4)
    print(s.dequeue())
    print(s.dequeue())
    print(s.dequeue())
    print(s.dequeue())