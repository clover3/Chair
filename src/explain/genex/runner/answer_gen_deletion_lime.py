import os
from typing import List

import numpy as np

from cache import load_from_pickle
from cpath import output_path
from explain.genex.load import PackedInstance, load_packed
from explain.genex.save_to_file import save_score_to_file, DropStop, Config2, ConfigShort


def main():
    for data_name in ["clue", "tdlt"]:
        for method in ["deletion", "lime"]:
            score_name = "{}_{}".format(data_name, method)
            for config in [DropStop, Config2, ConfigShort]:
                try:
                    save_name = "{}_{}.txt".format(score_name, config.name)
                    save_path = os.path.join(output_path, "genex", save_name)
                    scores: List[np.array] = load_from_pickle(score_name)
                    data: List[PackedInstance] = load_packed(data_name)
                    save_score_to_file(data, config, save_path, scores)
                except:
                    print(data_name)
                    raise



if __name__ == "__main__":
    main()
