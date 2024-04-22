



def binary_search(arr, key):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        print(left, mid, right)
        if arr[mid] == key:
            right = mid - 1
        elif arr[mid] < key:
            left = mid + 1
        else:
            right = mid - 1

    return left


def find_largest_smaller(arr, target):
    left = 0
    right = len(arr) - 1
    result = -1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] < target:
            result = mid
            left = mid + 1
        else:
            right = mid - 1

    return result



arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]

ret = find_largest_smaller(arr, 3.5)

print(ret, arr[ret])