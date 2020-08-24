import itertools
import json
import os
from typing import Dict, List


def dump_to_jsonl(output_path, data, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')
    print('Wrote {} records to {}'.format(len(data), output_path))


def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data


def enum_only_dir(root_dir):
    for dir_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_name)
        if os.path.isfile(dir_path):
            continue
        yield dir_name


def get_all_combinations(config_list: Dict[str, List]):
    config_sources = []
    keys = list(config_list.keys())
    keys.sort()
    for key in keys:
        item = config_list[key]
        cur_choices = []
        for v in item:
           cur_choices.append((key, v))
        config_sources.append(cur_choices)

    out_options = []
    for t in itertools.product(*config_sources):
        out_options.append(dict(t))

    return out_options
