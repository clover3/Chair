import pickle
import sys

def concat(path1, path2, path3):
    obj1 = pickle.load(open(path1, "rb"))
    obj2 = pickle.load(open(path2, "rb"))
    merge = obj1 + obj2
    pickle.dump(merge, open(path3, "wb"))

if __name__ == "__main__":
    path1 = sys.argv[1]
    path2 = sys.argv[2]
    path3 = sys.argv[3]
    concat(path1, path2, path3)
