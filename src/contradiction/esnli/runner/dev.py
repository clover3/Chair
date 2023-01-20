from typing import List

from contradiction.esnli.load_esnli import load_esnli
from contradiction.mnli_ex.nli_ex_common import NLIExEntry
from data_generator.NLI.enlidef import enli_tags


def main():
    for split in ["dev", "test"]:
        for tag_type in enli_tags:
            problems: List[NLIExEntry] = load_esnli(split, tag_type)
            print(f"{split} {tag_type} {len(problems)} items")


if __name__ == "__main__":
    main()