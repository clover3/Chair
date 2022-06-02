import json

from cpath import pjoin
from dataset_specific.cord.path_info import cord_dir


def load_queries():
    queries = json.load(open(pjoin(cord_dir, "queries.json"), "r"))
    return queries
