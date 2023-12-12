import sys

from omegaconf import OmegaConf

from cpath import output_path
from iter_util import load_jsonl, dump_to_jsonl
from misc_lib import path_join


# Filter out document that contains keyword other than the maker name
# which is related to the maker name
from trainer_v2.per_project.transparency.mmp.bias.common import contain_any


def load_car_related_non_generic():
    p = path_join(output_path, "mmp", "bias", "car_exp", "not_generic_car_keywords3.txt")
    keywords = [line.strip() for line in open(p, "r")]
    return keywords


def main():
    conf_path = sys.argv[1]
    config = OmegaConf.load(conf_path)
    keywords = load_car_related_non_generic()
    save_entrires = []
    for job_no in range(config.max_job):
        p = path_join(config.save_dir, str(job_no) + ".jsonl")
        relevant_entries = load_jsonl(p)

        def do_keep(item):
            text = item["doc_text"]
            return not contain_any(text, keywords)

        filtered_entries = list(filter(do_keep, relevant_entries))
        print(f"{len(relevant_entries)} -> {len(filtered_entries)}")
        save_entrires.extend(filtered_entries)

    dump_to_jsonl(config.filtered_save_dir, save_entrires)


if __name__ == "__main__":
    main()