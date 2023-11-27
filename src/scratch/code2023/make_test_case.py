

def make(arr):
    return ",".join(["[{}]".format(item) for item in arr])


def get_sequence(n):

    m = 2
    for i in range(n):
        yield m
        m = m * 2



print(make(get_sequence(20)))