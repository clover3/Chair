import sys


def main():
    template = sys.argv[1]
    st = int(sys.argv[2])
    ed = int(sys.argv[3])

    file_path_list = []
    for i in range(st, ed):
        if "{}" in template:
            file_path = template.format(i)
        else:
            file_path = template + str(i)
        file_path_list.append(file_path)

    print(",".join(file_path_list), end="")


if __name__ == "__main__":
    main()