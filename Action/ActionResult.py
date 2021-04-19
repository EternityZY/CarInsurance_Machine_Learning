import pickle
import os
import numpy as np
path = 'epoch_167.pkl'  #    pkl文件所在路径

f = open(path, 'rb')
data = pickle.load(f)

class_id = np.argmax(np.squeeze(np.array(data), axis=1), axis=1)

with open('category.txt') as f:
    lines = f.readlines()
categories = []
for line in lines:
    line = line.rstrip()
    categories.append(line)

test = []
with open('test.txt') as f:
    lines = f.readlines()
for line in lines:
    line = line.rstrip()
    test.append(line)

result_list = []
for i in range(len(data)):
    txt = test[i]
    id = class_id[i]
    res = txt + ' ' + categories[id]
    result_list.append(res)


with open('submission.txt', 'w') as f:
    f.write('\n'.join(result_list))
