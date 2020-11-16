import json

from exec_lib import run_func_with_config


def main(config):
    queries = json.load(open(config['query_path'], "r"))
    d = {}
    for entry in queries["queries"]:
        d[entry["number"]] = entry["text"]

    json.dump(d, open(config['save_path'], "w"))


if __name__ == "__main__":
    run_func_with_config(main)