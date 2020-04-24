import json

from cord.path_info import cord_dir
from cpath import pjoin


def load_queries():
    queries = json.load(open(pjoin(cord_dir, "queries.json"), "r"))
    return queries
