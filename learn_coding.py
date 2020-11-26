# -*- coding:utf-8 -*-
import sys
print(sys.stdout.encoding)
print(sys.getdefaultencoding())
# \xxx in str r
import sys

reload(sys)
sys.setdefaultencoding('utf8')

"""caption = "Aleksandar Kolarov, centre, fires home Manc\xa3hester City's third goal"
print(caption)
with open('test.txt', 'w') as f:
    f.write(caption)"""

"""a = "aaa\u2019aaa"
b = u'aa\u2019aa\xa3a'
print(type(a))
print(a)
print(type(b))
print(b)"""

# a = 'a\xa325b'
a_unicode = u'\u00a379m'
# a_unicode = u'aa\u2019a'
print(a_unicode)
print(type(a_unicode))
print(len(a_unicode))  # number of character
a_byte_str = a_unicode.encode('ascii', errors='ignore') # .decode('ascii')
print(type(a_byte_str))
print(len(a_byte_str))  # number of bytes
print(a_byte_str)
if '£' in a_unicode:
    with open('test.txt', 'w') as f:
        f.write(a_unicode)
    print('a in a_unicode')
