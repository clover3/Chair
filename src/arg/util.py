import json
import sys


def load_run_config():
    return json.load(open(sys.argv[1], "r"))
