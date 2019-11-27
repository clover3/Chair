
from functools import partial

class Dyn:
    def __init__(self):

        self.name = "my name"
        self.age = "my age"

        for attr in ["name", "age"]:
            def getter(target):
                return getattr(self, target)

            setattr(self, "get_"+attr, partial(getter, attr))






dyn = Dyn()

print(dyn.get_name())

print(dyn.get_age())


