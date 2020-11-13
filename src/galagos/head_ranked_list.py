import sys
from typing import List, Dict

from galagos.parse import load_galago_ranked_list, write_ranked_list_from_s
from galagos.types import GalagoDocRankEntry
from list_lib import dict_value_map


def main():
    file_path = sys.argv[1]
    top_n = int(sys.argv[2])
    save_path = sys.argv[3]
    ranked_list_d: Dict[str, List[GalagoDocRankEntry]] = load_galago_ranked_list(file_path)

    def get_head(l: List):
        return l[:top_n]

    new_ranked_list = dict_value_map(get_head, ranked_list_d)
    write_ranked_list_from_s(new_ranked_list, save_path)


if __name__ == "__main__":
    main()