class time_signature(object):
    def __init__(self, time, relation='NA', node_type='start'):
        self.time = extract(time)
        #       in each time point, the relation is a tuple indicates its before and after relationships
        self.relation = relation
        self.type = node_type

    def __str__(self):
        return (self.time, self.relation, self.type)

    #   so what does cmp do?
    def __cmp__(self, x):
        return cmp(self.time, x.time)

    #   less than
    def __lt__(self, x):
        if self.time == x.time:
            a = {'start': 0, 'mention': 1, 'end': 2}
            #           start and end point in same time
            if self.relation != x.relation:
                a = {'end': 0, 'start': 1, 'mention': 2}
            return a[self.type] < a[x.type]
        return self.time < x.time


def extract(time):
    if time is None:
        return (None, None, None)
    year = int(time[:4])
    month = int(time[5:7])
    day = int(time[8:])
    return year, month, day


class Mention(object):
    def __init__(self, sent, tag='NA', time=None, pos1=0, pos2=0):
        self.sent = sent
        self.time = time
        self.pos = (pos1, pos2)
        self.tag = tag

    def __lt__(self, x):
        return self.time < x.time



class Stack(object):
    # 初始化栈为空列表
    def __init__(self):
        self.items = []

    # 判断栈是否为空，返回布尔值
    def is_empty(self):
        return self.items == []

    # 返回栈顶元素
    def peek(self):
        return self.items[len(self.items) - 1]

    # 返回栈的大小
    def size(self):
        return len(self.items)

    def __getitem__(self, ix):
        return self.items[ix]

    # 把新的元素堆进栈里面（程序员喜欢把这个过程叫做压栈，入栈，进栈……）
    def push(self, item):
        self.items.append(item)

    # 把栈顶元素丢出去（程序员喜欢把这个过程叫做出栈……）
    def pop(self):
        return self.items.pop()

