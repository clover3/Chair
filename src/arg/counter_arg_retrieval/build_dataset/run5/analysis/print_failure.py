import os

from arg.counter_arg_retrieval.build_dataset.data_prep.remove_duplicate_passages import SplitDocDict
from arg.counter_arg_retrieval.build_dataset.run5.analysis.helper import load_ca_run_data
from arg.counter_arg_retrieval.build_dataset.run5.passage_scoring_util import load_run5_swtt_passage_as_d
from bert_api.swtt.segmentwise_tokenized_text import PassageSWTTUnit
from cpath import output_path, data_path, get_canonical_model_path
from data_generator.tokenizer_wo_tf import get_tokenizer, pretty_tokens
from trainer_v2.chair_logging import c_log

#for  TF 1.x
def main():
    voca_path = os.path.join(data_path, "bert_voca.txt")
    c_log.info("failure_analysis")
    c_log.info("Loading NLIEncoder")
    c_log.info("Init iter load_data")
    run_name = "PQ_12"
    entries = load_ca_run_data(run_name)
    tokenizer = get_tokenizer()

    passages_doc = {}

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
        print("Query:", pretty_tokens(e.q_tokens, True))
        print("Passages:", pretty_tokens(passage.get_as_subword(), True))


if __name__ == "__main__":
    main()