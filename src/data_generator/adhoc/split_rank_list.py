from data_generator.data_parser.robust2 import select_top_k_galago


def print_top_k_lines():
    for k in [100, 150]:
        f_out = open("rob04.galago.{}.out".format(k), "w")
        lines = select_top_k_galago(k)
        f_out.writelines(lines)


if __name__ == "__main__":
    print_top_k_lines()