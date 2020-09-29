import os
from typing import Iterable, Tuple, List

from arg.counter_arg.eval import pairwise_candidate_gen, EvalCondition
# use top-k candidate as payload
from arg.counter_arg.header import ArguDataPoint, Passage, splits
from arg.qck.data_util import write_qc_records
from arg.qck.decl import QCKQuery, QCKCandidate
from epath import job_man_dir
from list_lib import lmap
from misc_lib import exist_or_mkdir


def transform(t: Tuple[ArguDataPoint, Passage, bool]) -> Tuple[QCKQuery, QCKCandidate, bool]:
    problem, candidate, is_correct = t
    return QCKQuery(problem.text1.id.id, problem.text1.text), \
           QCKCandidate(candidate.id.id, candidate.text), \
           is_correct


def make_and_write(split, condition):
    records: Iterable[Tuple[ArguDataPoint, Passage, bool]] = pairwise_candidate_gen(split, condition)
    qc_records: List[Tuple[QCKQuery, QCKCandidate, bool]] = lmap(transform, records)
    dir_path = os.path.join(job_man_dir, "arg_bert")
    exist_or_mkdir(dir_path)
    output_path = os.path.join(dir_path, split)
    write_qc_records(output_path, qc_records)


def main():
    # transform payload to common QCK format
    for split in splits:
        print(split)
        make_and_write(split, EvalCondition.EntirePortalCounters)


if __name__ == "__main__":
    main()
