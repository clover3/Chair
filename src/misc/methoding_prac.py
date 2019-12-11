
class Foo:
    def __init__(self):
        self.var = 0


    def doit(self, v):
        print(v)
        pass



method_list = list([func for func in dir(Foo)
                         if callable(getattr(Foo, func))])


foo = Foo()
for method in method_list:
    if not method.startswith("__"):
        print(method)
        method_fn = getattr(Foo, method)
        method_fn(foo, 10)


