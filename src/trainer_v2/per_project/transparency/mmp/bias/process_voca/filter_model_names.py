from trainer_v2.per_project.transparency.mmp.bias.common import load_car_maker_list
from cpath import output_path
from misc_lib import path_join


def main():
    car_maker_base = load_car_maker_list()

    term_list_path = path_join(output_path, "mmp", "bias", "not_generic_car_keywords.txt")
    term_list = [line.lower().strip() for line in open(term_list_path, "r")]

    extended = []
    for term in term_list:
        for car_maker in car_maker_base:
            if term != car_maker and car_maker in term:
                print(f"{term}: {car_maker}")
                extended.append(term)
                break

    ex_term_list_path = path_join(output_path, "mmp", "bias", "car_maker_ex.txt")
    with open(ex_term_list_path, "w") as f:
        for t in extended:
            f.write(f"{t}\n")



if __name__ == "__main__":
    main()