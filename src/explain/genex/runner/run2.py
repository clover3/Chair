import os
from typing import List

from cpath import output_path
from explain.genex.load import PackedInstance, load_packed
from explain.genex.save_to_file import query_as_answer, \
    random_answer, save_score_to_file2


def main():
    for data_name in ["clue", "tdlt"]:
        for method in ["random", "query"]:
            try:
                save_name = "{}_{}.txt".format(data_name, method)
                save_path = os.path.join(output_path, "genex", save_name)
                data: List[PackedInstance] = load_packed(data_name)
                method_fn = {
                    "random": random_answer,
                    "query": query_as_answer
                }[method]
                save_score_to_file2(data, save_path, method_fn)
            except:
                print(data_name)
                raise


if __name__ == "__main__":
    main()

