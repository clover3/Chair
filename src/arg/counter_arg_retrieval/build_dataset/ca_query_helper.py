import csv
from typing import List

from arg.counter_arg_retrieval.build_dataset.ca_query import CATask


def load_ca_task_from_csv(file_path) -> List[CATask]:
    f = open(file_path)
    outputs: List[CATask] = []
    head = []
    for idx, row in enumerate(csv.reader(f)):
        if idx == 0:
            head = row
        else:
            d = {}
            for column, value in zip(head, row):
                d[column] = value

            e = CATask(d['qid'], d['conclusion'], d['c_stance'],
                       d['premise'], d['p_stance'], d['pc_stance'],
                       d['entity']
                       )
            outputs.append(e)
    return outputs