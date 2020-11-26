# -*- coding:utf-8 -*-
import numpy as np
from six.moves import cPickle
import sys
import os
import math
"""reload(sys)
sys.setdefaultencoding('utf8')"""

# '/' and '//'(向下取整)
"""print(9/5)
print(-9/5)
print(9 // 5)
print(-9//5)"""

# lambda + map
"""a = lambda x: (x!=0).sum()+2
print(a(np.array([1,3,2,0])))
b = np.array([[3,3,3,4,0],[9,8,4,0,0]])
print(list(map(a, b)))"""

# []*n
"""print([5]*6)
print([[2,2]]*4)
print([[2,2],[4,4]]*4)"""

# np.stack
"""a = [[1,2,3],[2,3,4],[5,6,7],[6,7,8]]
print(np.stack(a))
print(np.array(a))
print(np.stack(a, axis=1))
print(np.stack(a, axis=2))"""

# cpickle py3
"""infos={}
for i in range(3):
    infos['id'] = i
    infos['name'] = 'name_' + str(i)
    with open('test_cpickle.pkl', 'wb') as f:
        cPickle.dump(infos, f)  # overwrite previous object

with open('test_cpickle.pkl', 'rb') as f:
    infos = cPickle.load(f, encoding='iso-8859-1')  # encoding is not needed in py2
    print(infos)"""

# list.append(dict)
"""from copy import deepcopy
items =[]
item = {}
for i in range(5):
    item['id'] = i
    items.append(item)
    item = deepcopy(item)

print(items)"""

# \n in str
"""caption = "\n   Aleksandar Kolarov, centre, fires home \n Manchester City's third goal"
if '\n' in caption:
    caption = caption.strip().split('\n')[0].strip()
print(caption)"""

# pop
"""a = [1,1,1,1]
for i in range(7):
    a.pop()
    print('a')"""

# json dump
"""import  json
a = {'b':1}
json.dump(a, open('test.json', 'w'))
print(a)"""

# <*>
"""import re
caption = 'I <em>love</em> <p>china</p>.'
if '<' in caption and '</' in caption:
    print('###<X><\X>## in caption')
    caption = re.sub("<[^>]*>", '', caption)
    print(caption)"""

# sort
"""a = [{'x':1, 'y':2}, {'x':2, 'y':4}, {'x':4, 'y':1}]
a.sort(key=lambda e:e['y'], reverse=True)
print(a)"""

# truncate
"""a = "I love China."
start = 2
end = 6
a = a[0:start]+'verb'+a[end:]
print(a)"""

# list add
"""a = [1,2,3]
b = [2,3]
a+=b
print(a)"""

# str equal
"""dataset = 'breakingnews'
if 'breakingnews'==dataset:
    print('yes')"""
# dict summation
"""big_dict = {}
def create_dict(id):
    small_dict = {}
    small_dict[id]=str(id*10)
    return small_dict
for i in range(5):
    small_dict = create_dict(i)
    big_dict = dict(big_dict.items()+small_dict.items())
print(big_dict)"""

# file exist
# print(os.path.exists("file/test.txt"))

# string split
"""filename = "data/07/25/image/n199600702a_i0.jpg"
img_id = filename.split('/')[-1].replace('.jpg', '')
print(img_id)"""

# dict.getkeys()
"""A = {'a':1, 'b':2, 'c':3}
B = {k: A[k]+1 for k in A.keys()}
print(A.keys())
print(B.keys())
assert A.keys() == B.keys()"""

# use str as bool
"""sent_emb = ''
if sent_emb:
    print('sent emb')"""

# groupby
"""from itertools import groupby
temp = ['PERSON_', 'PERSON_', 'is', 'a', 'teacher', 'in', 'ORG_', 'ORG_', 'ORG_']
print(groupby(temp))
template = [x[0] for x in groupby(temp)]
print(template)"""

# count
""""from collections import Counter
c= Counter('aaaaaaaaaaaaaaaaa')
print(c.get('a', 5))"""

# sort
# a = {'person':5, 'location':8, 'org':2}
"""a = {1:0.0, 2:0.75}
sorted_a = sorted([(en_type, count) for en_type, count in a.items()], reverse=True, key=lambda e:e[1])
for item in sorted_a:
    print(item)"""
"""a = [3,7,1,2,0]
a.sort(key=lambda x: x, reverse=False)
print(a)"""

# log
"""a = math.log(4, 2)
print(a)"""

#
"""W = 7.8489
tf = 0
K = 2.0
k1 = 2.0
k2 = 1.0
qf = 1
token_score = (W * tf * (k1 + 1) / (tf + K)) * (qf * (k2 + 1) / (qf + k2))
print(token_score)"""

"""a = [[1,2,3], [4,5,6]]
b= []
for x in a:
    b+= x
b =b[:6]
print(b)"""

"""a = [[0]*5]*4
c = np.array([a]*3)
print(c)
print(c[:, :2])"""

"""answer_word_matches = [[6], [30], [22], [63, 7284], [64, 7288, 7289], [24, 7286], [72, 7287], [73], [74, 7285], [7293]]
max_match_num= 20
idx_seq_list = [()]
for matched_inds in answer_word_matches:
    idx_seq_list = [
        seq + (idx,)
        for seq in idx_seq_list for idx in matched_inds
    ]
    if len(idx_seq_list) > max_match_num:
        idx_seq_list = idx_seq_list[:max_match_num]
    print('id_seq_list:', idx_seq_list)"""

a = [1,2,3,4,5,6,7,8,9]
print(a[1:-1])






