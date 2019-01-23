import torch

import collections
iter_str = 'abcdefg'
#创建一个OrderedDict的实例index
index = collections.OrderedDict()
#把字符串列表添加到index集合里面
for i in range(7) :
    index[i] = list(iter_str)[i]
print(index)
#删除index的第一个key-value
index.popitem(last = False)
print(index)
print(index[3])
