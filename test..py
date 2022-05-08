import os


class DaShaMei(object):
    def __init__(self, num):
        self.nnum = num
        print('a')


    def leijia(self):
        a = 0
        for i in range(self.nnum):
            a += i
        return a

    def dameidarenle(self):
        print('hit me!')


class Dacongge(DaShaMei):
    def __init__(self):
        super(Dacongge, self).__init__(num=1)
        self.age = 18


dcg = Dacongge()
print(dcg.age)
dcg.dameidarenle()

exit()




dsm = DaShaMei(num=1099)
e = dsm.leijia()
print(e)
# dsm.dameidarenle()
# print(dsm.a)

# dsm2 = DaShaMei()
# print(dsm2.a)
c = []
for i in range(100):
    b = DaShaMei(num=i)
    print(type(b))
    print(type(2))
    print(b.nnum)
    c.append(b)
print(len(c), c)


class dict2(object):
    def __init__(self):
        self.b = []

    def append(self, data):
        self.b += data


a = list()
print(type(a))
print(a)


print('x\nxx', end='\n')
print('\n', end='\n')
print('bbb')



