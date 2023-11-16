import pickle
from collections import namedtuple
from typing import NamedTuple


BaseTuple = namedtuple('BaseTuple', 'name')
class MyNamedTuple(BaseTuple):
    def __new__(cls, name="clover"):
        return super(MyNamedTuple, cls).__new__(cls, name)

    def __reduce__(self):
        # Return a callable and its arguments to recreate the object
        return (recreate_my_named_tuple, (self.name,))


def recreate_my_named_tuple(name):
    # This function will be called when unpickling the object
    return MyNamedTuple(name)


def main():
    obj = MyNamedTuple()
    pickle.dump(obj, open("/tmp/pickle", "wb"))


if __name__ == "__main__":
    main()