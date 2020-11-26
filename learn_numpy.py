import numpy as np

"""a = [1.0,2.0]
b = np.array(a)
print(b, type(b))"""

"""labels_batch = np.zeros([3,2])
labels = labels_batch[:, 1]
print(labels)"""

"""a = np.zeros([1,3])
print(a)"""

"""a = np.array([[1,1,1],[2,2,2],[3,3,3]])
b = np.pad(a,((0, 0),(0,0)),'constant', constant_values=0)
print(b)
sent_data = []
max_pointers_num = 8
article_data = np.zeros([max_pointers_num, 4])
if len(sent_data) > 0:
    pointer_num = len(sent_data)
    sent_data = np.pad(np.array(sent_data), (0, max_pointers_num - pointer_num), 'constant', constant_values=0)
    article_data[:, 2] = sent_data
print(article_data)"""

# reshape
"""temp = np.zeros([3, 4])
temp[:, 0] = np.array([0, 0, 0])
temp[:, 1] = np.array([1, 1, 1])
temp[:, 2] = np.array([2, 2, 2])
temp[:, 3] = np.array([3, 3, 3])
print(temp)
a = np.array([[0,1,2,3],[0,1,2,3],[0,1,2,3]]) # 3*4
b = a.reshape(-1, 3)
b = [i for d in b for i in d]
print(b)"""

"""sen_ids = np.zeros(7, dtype='uint32')
print(sen_ids)"""

# where sort argsort
"""d = np.array([[5,9,2,3,7,8]]).flatten()
print(d)
sort_d = np.sort(d)[-5:][::-1]
print(sort_d)
inds = np.argsort(d)
print(inds, inds[-5:])
inds = inds[-5:][::-1]
print(inds.tolist())
tmp = np.where(inds == 3)
print(tmp)
print(tmp[0][0])"""

"""a = np.array([[0,1,2,0,0],[0,1,0,0,0],[0,2,3,4,0]])
mask_batch = np.zeros([3, 5], dtype='float32')
nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, a)))
print(nonzeros)
for ix, row in enumerate(mask_batch):
    row[:nonzeros[ix]] = 1
print(mask_batch)"""

"""a = np.array([0,0,0,9,0,0])
print(a.shape)
a = a - 1
print(a)
b = [1 if x > 0 else 0 for x in a]
print(b)
print(np.array(b))"""

"""a = np.random.uniform(-1,1,size=[5,4])
a[0]= 0
print(a)"""

d = np.array([5,9,2,3,7,8])
a = np.argsort(d)[::-1][:7]
print(a)