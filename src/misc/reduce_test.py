from list_lib import lreduce, unique_from_sorted

l = [1, 1, 2, 2, 3, 3, 3]

def combine(prev_list, new_elem):
    if not prev_list or prev_list[-1] != new_elem:
        return prev_list + [new_elem]
    else:
        return prev_list



print(lreduce([], combine, l))
print(unique_from_sorted(l))