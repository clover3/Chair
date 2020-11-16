import json
import os

from arg.counter_arg.eval import load_problems
from arg.counter_arg.header import splits
from cpath import output_path


def main():
    problems = []
    for split in splits:
        problems.extend(load_problems(split))

    d = {}
    for p in problems:
        d[p.text1.id.id] = p.text1.text

    save_path = os.path.join(output_path, "text_d.json")
    print(save_path)
    json.dump(d, open(save_path, "w"))


if __name__ == "__main__":
    main()

