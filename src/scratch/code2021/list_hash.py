from misc_lib import int_list_to_str, recover_int_list_str

s = set()
d = {}

key = [1,2,3]

d[int_list_to_str(key)] = 1

for key in d:

    print(key)
    print(recover_int_list_str(key))