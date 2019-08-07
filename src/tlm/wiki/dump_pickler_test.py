from cache import DumpPickle, DumpPickleLoader

def do():
    dp = DumpPickle("temp.test")

    arr1 = list([j for j in range(10)])
    arr2 = list([j for j in range(20,30)])

    dp.dump("arr1", arr1)
    dp.dump("arr2", arr2)

    dp.close()


    dpl = DumpPickleLoader("temp.test")

    print(dpl.load("arr1"))
    print(dpl.load("arr2"))



do()
