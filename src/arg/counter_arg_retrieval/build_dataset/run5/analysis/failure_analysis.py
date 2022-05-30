import os
from typing import List, Tuple, NamedTuple

from arg.counter_arg_retrieval.build_dataset.data_prep.remove_duplicate_passages import SplitDocDict
from arg.counter_arg_retrieval.build_dataset.path_helper import load_sliced_passage_ranked_list
from arg.counter_arg_retrieval.build_dataset.run3.swtt.nli_common import EncoderForNLI
from arg.counter_arg_retrieval.build_dataset.run5.passage_scoring_util import load_run5_swtt_passage_as_d
from arg.counter_arg_retrieval.build_dataset.run5.path_helper import load_premise_queries
from bert_api.swtt.segmentwise_tokenized_text import PassageSWTTUnit
from cpath import output_path, data_path, get_canonical_model_path
from data_generator.bert_input_splitter import split_p_h_with_input_ids
from data_generator.shared_setting import BertNLI
from data_generator.tokenizer_wo_tf import get_tokenizer
from explain.nli_ex_predictor import NLIExPredictor
from misc_lib import two_digit_float
from models.transformer import hyperparams
from tlm.token_utils import cells_from_tokens
from trainer.np_modules import get_batches_ex
from trainer.promise import PromiseKeeper, MyPromise
from trainer_v2.chair_logging import c_log
from trec.qrel_parse import load_qrels_structured
from visualize.html_visual import HtmlVisualizer, Cell, normalize100


def join_ranked_list_w_qrel(qrels, run_name):
    pq = load_sliced_passage_ranked_list(run_name)
    for qid, entries in pq.items():
        try:
            qrel_per_qid = qrels[qid]
            for e in entries:
                try:
                    judged_score = qrel_per_qid[e.doc_id]
                    yield qid, e.doc_id, e.score, judged_score
                except KeyError:
                    pass
        except KeyError:
            pass


class _Entry(NamedTuple):
    qid: str
    q_tokens: List[str]
    passage_id: str
    model_score: float
    judged_score: int


def load_data() -> List[_Entry]:
    # List[(qid, passage_id, passage_content, model score, judged score)]
    tokenizer = get_tokenizer()
    query_list: List[Tuple[str, str]] = load_premise_queries()
    query_d = dict(query_list)

    judgment_path = os.path.join(output_path, "ca_building", "qrel", "0522.txt")
    qrels = load_qrels_structured(judgment_path)
    run_name = "PQ_12"
    for qid, passage_id, model_score, judged_score in join_ranked_list_w_qrel(qrels, run_name):
        query_text = query_d[qid]
        q_tokens = tokenizer.tokenize(query_text)
        e = _Entry(qid,
                   q_tokens,
                   passage_id,
                   model_score,
                   judged_score
               )
        yield e


def get_senli():
    hp = hyperparams.HPSENLI3()
    nli_setting = BertNLI()

    model_save_path = os.path.join(get_canonical_model_path("nli_ex_21"), "model-73630")
    modeling_option = "CE"
    predictor = NLIExPredictor(hp, nli_setting, model_save_path, modeling_option)
    return predictor

#for  TF 1.x
def main():
    runs = ["PQ_10", "PQ_11", "PQ_12", "PQ_13"]
    voca_path = os.path.join(data_path, "bert_voca.txt")
    c_log.info("failure_analysis")
    c_log.info("Loading NLIEncoder")
    encoder = EncoderForNLI(300, voca_path)
    c_log.info("Init iter load_data")
    entries = load_data()
    c_log.info("get_senli()")
    senli: NLIExPredictor = get_senli()
    tokenizer = get_tokenizer()
    batch_size = 2
    c_log.info("start iterations()")

    def solve_fn(x_list):
        batches = get_batches_ex(x_list, batch_size, 3)
        ret = senli.predict_both("conflict", batches)
        sout, ex_logits = ret
        return [(sout[idx], ex_logits[idx]) for idx in range(len(x_list))]

    passages_doc = {}

    pk = PromiseKeeper(solve_fn)
    e_out_list = []
    for e in entries:
        is_fp = e.model_score > 0.5 and e.judged_score == 0
        is_fn = e.model_score < 0.5 and e.judged_score == 1
        intersting = is_fp or is_fn
        if not intersting:
            continue
        if e.qid not in passages_doc:
            passages_doc[e.qid] = load_run5_swtt_passage_as_d(e.qid)
        passages: SplitDocDict = passages_doc[e.qid]
        doc_id, passage_idx = e.passage_id.split("_")
        swtt, passage_ranges = passages[doc_id]
        passage = PassageSWTTUnit(swtt, passage_ranges, int(passage_idx))

        triplet = encoder.encode_token_pairs(e.q_tokens, passage.get_as_subword())
        promise = MyPromise(triplet, pk)
        e_out = (e, triplet, promise)
        e_out_list.append(e_out)

    pk.do_duty(True)
    html = HtmlVisualizer("senli_ca_search.html")
    for e_out in e_out_list:
        e, triplet, promise = e_out
        probs, ex_logit = promise.future().get()
        input_ids, _, _ = triplet
        p, h = split_p_h_with_input_ids(input_ids, input_ids)
        p_score, h_score = split_p_h_with_input_ids(ex_logit, input_ids)

        def format_token_scores(scores, input_ids):
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            out_s_tokens = ["{0}({1:.2f})".format(token, score) for score, token in zip(scores, tokens)]
            out_s = " ".join(out_s_tokens)
            return out_s

        def format_token_cells(scores, input_ids) -> List[Cell]:
            max_v = max(scores)
            scores = [normalize100(s, max_v) for s in scores]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            return cells_from_tokens(tokens, scores)
            # out_s_tokens: List[Cell] = [Cell(token, score) for score, token in zip(scores, tokens)]
            # return out_s_tokens
        decision_str = "gold={0} p={1:.2f}".format(e.judged_score, e.model_score)

        pred_str ="Prediction: " + ",".join(map(two_digit_float, probs))
        html.write_paragraph(decision_str)
        html.write_paragraph(pred_str)
        html.write_paragraph("Document: ")
        html.write_table([format_token_cells(p_score, p)])
        html.write_paragraph("Premise: ")
        html.write_table([format_token_cells(h_score, h)])
        html.write_bar()


if __name__ == "__main__":
    main()