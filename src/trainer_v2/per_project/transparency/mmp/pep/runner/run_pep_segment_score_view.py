from typing import List, Iterable, Callable, Dict, Tuple, Set


import sys

from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import lflatten, lmap
from misc_lib import get_second, two_digit_float, TimeEstimator
from table_lib import tsv_iter
from trainer.promise import PromiseKeeper, MyFuture, list_future
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.definitions import ModelConfig512_1
from trainer_v2.per_project.transparency.mmp.pep.demo_util import get_pep_local_decision, PEPLocalDecision
from trainer_v2.train_util.get_tpu_strategy import get_strategy
from visualize.html_visual import HtmlVisualizer, Cell


def mask_remain(indices, sb_tokens) -> List[str]:
    out_sb_tokens = []
    for i, t in enumerate(sb_tokens):
        if i in indices:
            out_sb_tokens.append(t)
        else:
            t = ["[MASK]"] * len(t)
            out_sb_tokens.append(t)
    return lflatten(out_sb_tokens)


QD = Tuple[str, str]
def check_full_pairwise_tokens(
        model_path,
         qd_pair_iter: Iterable[Tuple[QD, QD]]):
    tokenizer = get_tokenizer()
    sp_tokenize = tokenizer.basic_tokenizer.tokenize
    sb_tokenize = tokenizer.wordpiece_tokenizer.tokenize
    strategy = get_strategy()
    model_config = ModelConfig512_1()

    with strategy.scope():
        pep = PEPLocalDecision(model_config, model_path)

    def analyze_qd(qd: QD):
        q, d = qd
        q_sp_tokens: list[str] = sp_tokenize(q)
        q_sb_tokens: List[List[str]] = lmap(sb_tokenize, q_sp_tokens)
        d_sp_tokens: list[str] = sp_tokenize(d)
        d_sb_tokens: List[List[str]] = lmap(sb_tokenize, d_sp_tokens)

        output_future: List[List[MyFuture]] = []
        pk = PromiseKeeper(pep.score_fn)
        c_log.info("Work for qd")
        for q_idx, q_sp_token in enumerate(q_sp_tokens):
            row: List[MyFuture] = []
            for d_idx, d_sp_token in enumerate(d_sp_tokens):
                q_masked: List[str] = mask_remain([q_idx], q_sb_tokens)
                d_masked: List[str] = mask_remain([d_idx], d_sb_tokens)
                score_future = pk.get_future((q_masked, d_masked))
                row.append(score_future)
            output_future.append(row)

        pk.do_duty()
        output = [list_future(row) for row in output_future]
        c_log.info("done")

        return q_sp_tokens, d_sp_tokens, output

    for qd_pair in qd_pair_iter:
        qd_pos, _qd_neg = qd_pair
        yield analyze_qd(qd_pos)


def main():
    itr = tsv_iter(sys.argv[1])

    def qd_iter():
        for q, d1, d2 in itr:
            yield (q, d1), (q, d2)

    model_path = sys.argv[2]
    analyzed_iter = check_full_pairwise_tokens(model_path, qd_iter())

    html = HtmlVisualizer("pep_segment_sel.html")

    n_out = 0
    for q_sp_tokens, d_sp_tokens, output in analyzed_iter:
        table = []
        head = [""] + d_sp_tokens
        for q_term, scores in zip(q_sp_tokens, output):
            assert len(scores) == len(d_sp_tokens)

            def norm(score):
                norm_score = int(min(abs(score) * 100, 100))
                return norm_score

            def get_color(score):
                color = "B" if score > 0 else "R"
                return color

            row_text = [q_term] + lmap(two_digit_float, scores)
            row_scores = [0] + lmap(norm, scores)
            row_colors = ["R"] + lmap(get_color, scores)

            row = []
            for t, s, c in zip(row_text, row_scores, row_colors):
                row.append(Cell(t, s, target_color=c))

            table.append(row)

        html.write_table(table, head=head)

        n_out += n_out
        if n_out > 10:
            break



if __name__ == "__main__":
    main()