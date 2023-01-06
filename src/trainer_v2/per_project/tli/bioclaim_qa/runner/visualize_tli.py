
from typing import List, Iterable, Callable, Dict, Tuple, Set

from contradiction.medical_claims.annotation_1.load_data import load_reviews_for_split
from contradiction.medical_claims.load_corpus import Review
from dataset_specific.scitail import ScitailEntry, load_scitail_structured
from misc_lib import average
from trainer_v2.keras_server.name_short_cuts import get_pep_cache_client
from trainer_v2.per_project.tli.bioclaim_qa.eval_helper import get_bioclaim_retrieval_corpus
from trainer_v2.per_project.tli.enum_subseq import enum_subseq_136
from trainer_v2.per_project.tli.tli_visualize import til_to_table
from trainer_v2.per_project.tli.token_level_inference import TokenLevelInference, Numpy2D, \
    nc_max_e_avg_reduce_then_softmax, nc_max_e_avg_reduce_then_norm

from visualize.html_visual import Cell, HtmlVisualizer


def main():
    split = "dev"
    queries, claims = get_bioclaim_retrieval_corpus(split)
    review_list: List[Tuple[int, Review]] = load_reviews_for_split(split)
    qrel = {}
    for group_no, r in review_list:
        qid = str(group_no)
        for c in r.claim_list:
            doc_id = c.pmid
            qrel[qid, doc_id] = 1

    nli_predict_fn = get_pep_cache_client()
    tli_module = TokenLevelInference(nli_predict_fn, enum_subseq_136)

    html = HtmlVisualizer("bioclaim_tnli.html")
    for qid, query in queries:
        n_neg = 0
        n_true = 0
        todo = []
        payload = []
        for doc_id, claim in claims:
            label = "None"
            if (qid, doc_id) in qrel and qrel[qid, doc_id]:
                do_print = True
                n_true += 1
                label = "True"
            elif n_neg < n_true:
                do_print = True
                n_neg += 1
                label = "False"
            else:
                do_print = False

            if do_print:
                payload.append((claim, query))
                todo.append((qid, doc_id, label))

        tli_d: Dict[Tuple[str, str], Numpy2D] = tli_module.do_batch_return_dict(payload)
        for i in range(len(payload)):
            c, q = payload[i]
            tli: Numpy2D = tli_d[c, q]
            table: List[List[Cell]] = til_to_table(q, tli)
            _, _, label = todo[i]
            e_sum = sum(tli[:, 0])
            e_avg = average(tli[:, 0])
            probs = nc_max_e_avg_reduce_then_norm(tli)

            html.write_paragraph("Claim 1: " + c)
            html.write_paragraph("Question : " + q)
            html.write_paragraph("Gold : " + label)
            html.write_paragraph("sum={0:2f}, avg={1:.2f}, probs={2}".format(e_sum, e_avg, probs))
            html.write_table(table)
            html.write_bar()


if __name__ == "__main__":
    main()