
from collections import Counter
from typing import List, Iterable, Tuple
from typing import NamedTuple

from arg.counter_arg.eval import get_eval_payload_from_dp, prepare_eval_data
from arg.counter_arg.header import Passage
from arg.perspectives.pc_tokenizer import PCTokenizer
from list_lib import left, foreach
from list_lib import lmap
from models.classic.lm_util import tokens_to_freq, average_counters, get_lm_log, smooth, subtract, least_common


class RelevanceModel(NamedTuple):
    query_id: str
    text: str
    lm: Counter


def build_lm(split) -> Iterable[RelevanceModel]:
    tokenizer = PCTokenizer()
    problems, candidate_pool_d = prepare_eval_data(split)
    payload: List[Passage] = get_eval_payload_from_dp(problems)
    for query, problem in zip(payload, problems):
        p = problem
        source_text = p.text1.text
        tokens = tokenizer.tokenize_stem(source_text)
        counter = tokens_to_freq(tokens)
        yield RelevanceModel(query.id.id, query.text, counter)


if __name__ == "__main__":
    split = "training"
    lms: List[Tuple[str, Counter]] = list(build_lm(split))
    alpha = 0.1
    bg_lm = average_counters(lmap(lambda x: x.lm, lms))

    def show(r: RelevanceModel):
        print('----')
        print(r.text)
        log_topic_lm = get_lm_log(smooth(r.lm, bg_lm, alpha))
        log_bg_lm = get_lm_log(bg_lm)
        log_odd: Counter = subtract(log_topic_lm, log_bg_lm)

        for k, v in r.lm.most_common(50):
            print(k, v)

        s = "\t".join(left(r.lm.most_common(10)))
        print("LM freq: ", s)
        print(s)

        s = "\t".join(left(log_odd.most_common(30)))
        print("Log odd top", s)

        s = "\t".join(left(least_common(log_odd, 10)))
        print("Log odd bottom", s)

    foreach(show, lms[:10])