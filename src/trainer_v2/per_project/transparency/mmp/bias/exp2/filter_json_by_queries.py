import sys

from omegaconf import OmegaConf

from cache import save_list_to_jsonl
from iter_util import load_jsonl
from trainer_v2.per_project.transparency.misc_common import read_lines


def main():

    conf_path = sys.argv[1]
    config = OmegaConf.load(conf_path)

    relevant_entries = load_jsonl(config.entry_jsonl_path)
    queries = read_lines(config.manual_filtered_queries)

    output = []
    for entry in relevant_entries:
        if entry["query"] in queries:
            output.append(entry)

    save_list_to_jsonl(output, config.filtered_jsonl_save_path)



if __name__ == "__main__":
    main()