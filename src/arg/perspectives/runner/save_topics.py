from typing import List, Dict

from arg.perspectives.load import load_claim_perspective_pair
from cpath import at_data_dir


def main():
    pc_data: List[Dict] = load_claim_perspective_pair()

    out_f = open(at_data_dir("perspective", "claims.txt"), "w")

    for e in pc_data:
        cid = e['cId']
        text = e['text']
        row = [str(cid), text]
        out_f.write("\t".join(row) + "\n")


if __name__ == "__main__":
    main()