
class SomeClass:
    def __init__(self, l):
        self.l = l

    def __hash__(self):
        return str(self.l)



d = set()
obj = SomeClass([1])
obj2 = SomeClass([1])
d.add(obj)

print(obj in d)
print(obj2 in d)