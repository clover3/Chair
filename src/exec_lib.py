import json
import sys


def run_func_with_config(fn):
    config = json.load(open(sys.argv[1], "r"))
    fn(config)
