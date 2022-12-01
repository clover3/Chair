from typing import List

from dataset_specific.ists.parse import iSTSProblemWChunk
from dataset_specific.ists.path_helper import load_ists_problems_w_chunk
from dataset_specific.ists.split_info import ists_enum_split_genre_combs

for split, genre in ists_enum_split_genre_combs():
    problems: List[iSTSProblemWChunk] = load_ists_problems_w_chunk(genre, split)
