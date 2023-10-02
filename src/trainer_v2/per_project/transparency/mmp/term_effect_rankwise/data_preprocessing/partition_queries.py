from typing import List, Iterable, Tuple

from dataset_specific.msmarco.passage.passage_resource_loader import enum_grouped, FourItem
from table_lib import tsv_iter
from dataset_specific.msmarco.passage.path_helper import get_mmp_grouped_sorted_path

# Output (Doc_id, TFs Counter, base score)
from trainer_v2.per_project.transparency.misc_common import save_tsv
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import get_grouped_queries_path
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_valid_mmp_partition_for_train


def work_for(itr: Iterable[List[FourItem]]) -> Iterable[Tuple[str, str]]:
    for group in itr:
        for qid, pid, query, text in group:
            yield qid, query
            break


def main():
    split = "train"
    for job_no in get_valid_mmp_partition_for_train():
        print(job_no)
        itr = tsv_iter(get_mmp_grouped_sorted_path(split, job_no))
        g_itr: Iterable[List[FourItem]] = enum_grouped(itr)
        qid_text_itr = work_for(g_itr)
        save_path = get_grouped_queries_path(split, job_no)
        save_tsv(qid_text_itr, save_path)


if __name__ == "__main__":
    main()