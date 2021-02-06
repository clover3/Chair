import os
from typing import List

import numpy as np

from cache import load_pickle_from
from cpath import output_path, data_path
from explain.genex.load import PackedInstance, load_packed
from explain.genex.save_to_file import save_score_to_file, DropStop, Config2, ConfigShort


def main():
    data_name = "wiki"
    for method in ["deletion", "LIME"]:
        for config in [DropStop, Config2, ConfigShort]:
            data_method_str = "{}_{}".format(data_name, method)
            save_dir = os.path.join(output_path, "genex", data_method_str)
            for i in range(100):
                try:
                    idx_str = "{0:02d}".format(i)
                    score_name = "{}_{}_{}".format(data_name, idx_str, method)
                    save_name = "{}_{}.txt".format(score_name, config.name)
                    save_path = os.path.join(save_dir, save_name)
                    score_path = os.path.join(data_path, "cache", data_method_str, score_name + ".pickle")
                    scores: List[np.array] = load_pickle_from(score_path)
                    data: List[PackedInstance] = load_packed(data_name)
                    save_score_to_file(data, config, save_path, scores)
                except:
                    print(data_name)
                    raise



if __name__ == "__main__":
    main()
