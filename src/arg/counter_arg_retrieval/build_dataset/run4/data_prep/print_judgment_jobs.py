import os
from typing import Tuple, Dict, List

from arg.counter_arg_retrieval.build_dataset.annotation_prep import save_judgement_entries
from arg.counter_arg_retrieval.build_dataset.passage_scoring.split_passages import PassageRange
from arg.counter_arg_retrieval.build_dataset.run4.data_prep.get_judgments_todo import get_judgments_todo
from arg.counter_arg_retrieval.build_dataset.run4.run4_util import load_run4_swtt_passage, load_ca4_tasks
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from cpath import output_path
from trec.types import DocID


def main():
    todo = get_judgments_todo()
    passages: Dict[DocID, Tuple[SegmentwiseTokenizedText, List[PassageRange]]] \
        = load_run4_swtt_passage()
    ca_tasks = load_ca4_tasks()
    save_path = os.path.join(output_path, "ca_building", "run4", "annot_jobs.csv")
    save_judgement_entries(todo,
                           passages,
                           ca_tasks,
                           save_path)


if __name__ == "__main__":
    main()
