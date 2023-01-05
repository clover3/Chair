from typing import List

from dataset_specific.scitail import ScitailEntry, load_scitail_structured
from trainer_v2.per_project.tli.scitail_qa_eval.path_helper import get_qrel_path
from trainer_v2.per_project.tli.scitail_qa_eval.runner.convert_to_ranked_list import build_qid_d
from trec.trec_parse import write_trec_relevance_judgement
from trec.types import TrecRelevanceJudgementEntry


def build_save_qrel(split):
    problems = load_scitail_structured(split)
    qid_d = build_qid_d(split)

    rel_entries = []
    for idx, problem in enumerate(problems):
        qid = qid_d[problem.question]
        label = problem.get_relevance_label()
        doc_id = f"{split}_{idx}"
        e = TrecRelevanceJudgementEntry(qid, doc_id, label)
        rel_entries.append(e)

    write_trec_relevance_judgement(rel_entries, get_qrel_path(split))


def main():
    for split in ["dev", "test"]:
        build_save_qrel(split)


if __name__ == "__main__":
    main()