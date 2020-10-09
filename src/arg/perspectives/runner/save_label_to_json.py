import json
from typing import List, Dict

from arg.perspectives.load import get_claim_perspective_id_dict2


def main():
    cid_to_pids: Dict[int, List[int]] = get_claim_perspective_id_dict2()

    out_data = []
    for cid, pid_list in cid_to_pids.items():
        for pid in pid_list:
            out_data.append((cid, pid, 1))

    json.dump(out_data, open("data/perspective/label_d.json", "w"))


if __name__ == "__main__":
    main()