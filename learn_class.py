class A(object):
    def __init__(self, name, gender, age):
        print('A init')
        self.namea = 'A' + name
        self.gendera = 'A' + gender

    def funca(self):
        print('name a', self.namea)


class B(A):
    def __int__(self, name, gender, age):
        super(B, self).__init__(name, gender, age)
        print('B init')
        self.nameb = 'B' + name
        self.genderb = 'B' + gender
        self.ageb = 'B' + age

    def funcb(self):
        print('name b', self.ageb)


if __name__ == '__main__':
    b = B(name='anwen', gender='male', age='25')
    b.funca()
    print(b.ageb)
