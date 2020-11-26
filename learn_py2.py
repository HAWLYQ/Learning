import difflib

# enumerate
"""a = enumerate([1,2,3,4])
# py2 .next; py3 .__next__
_, values = a.__next__()
print(values)
for i, v in a:
    print(i, v)"""

# range
# py2 range return list; py3:list(range())
"""istarts = [0] * 6+ list(range(10 - 6 + 1))
print(istarts)"""


# list +=
a = [1,2,3,4]
a += [0]*4
print(a)


