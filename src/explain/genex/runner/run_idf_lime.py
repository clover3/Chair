import sys
from typing import List

from cache import save_to_pickle
from explain.genex.idf_lime import explain_by_lime_idf, load_idf_fn_for
from explain.genex.load import load_as_lines
from misc_lib import tprint


def main():
    method = "idflime"
    data_name = sys.argv[1]
    try:
        save_name = "{}_{}".format(data_name, method)
        data: List[str] = load_as_lines(data_name)

        tprint("Loading idf")
        get_idf = load_idf_fn_for(data_name)
        tprint("Evaluating lime ")
        explains = explain_by_lime_idf(data, get_idf)
        save_to_pickle(explains, save_name)
    except:
        print(data_name)
        raise


if __name__ == "__main__":
    main()

