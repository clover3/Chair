import os
from typing import Dict

from arg.perspectives.load import get_perspective_dict
from cpath import output_path
from trec.trec_parse import trec_writer


def main():
    d: Dict[str, str] = get_perspective_dict()

    save_path = os.path.join(output_path, "perspective", "corpus.xml")
    f = open(save_path, "w")
    for pid, text in d.items():
        lines = trec_writer(pid, text)
        f.writelines(lines)
    f.close()



if __name__ == "__main__":
    main()