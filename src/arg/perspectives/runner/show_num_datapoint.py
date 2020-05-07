from typing import List

from arg.perspectives.basic_analysis import load_data_point, PerspectiveCandidate


def show():
    for split in ["train", "dev"]:
        data_points: List[PerspectiveCandidate] = load_data_point(split)
        print(split, len(data_points))


show()
