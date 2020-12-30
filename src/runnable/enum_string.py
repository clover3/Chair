import sys


def main():
    prefix = sys.argv[1]
    st = int(sys.argv[2])
    ed = int(sys.argv[3])

    file_path_list = []
    for i in range(st, ed):
        file_path = prefix + str(i)
        file_path_list.append(file_path)

    print(",".join(file_path_list), end="")


if __name__ == "__main__":
    main()