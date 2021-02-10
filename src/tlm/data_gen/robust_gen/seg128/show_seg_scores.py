import os
from typing import List, Dict, Tuple

import numpy as np

from arg.qck.decl import get_format_handler
from arg.qck.prediction_reader import load_combine_info_jsons
from cpath import at_output_dir, output_path
from data_generator.data_parser.robust import load_robust_04_query
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import find_where, lmap
from misc_lib import two_digit_float
from tlm.data_gen.robust_gen.seg_lib.seg_score_common import OutputViewer, get_probs, get_piece_scores, ScoredPiece
from tlm.robust.load import get_robust_qid_list
from visualize.html_visual import HtmlVisualizer, Cell


def main():
    n_factor = 16
    step_size = 16
    max_seq_length = 128
    max_seq_length2 = 128 - 16
    batch_size = 8
    info_file_path = at_output_dir("robust", "seg_info")
    queries = load_robust_04_query("desc")
    qid_list = get_robust_qid_list()

    f_handler = get_format_handler("qc")
    info: Dict = load_combine_info_jsons(info_file_path, f_handler.get_mapping(), f_handler.drop_kdp())
    print(len(info))
    tokenizer = get_tokenizer()

    for job_idx in [1]:
        qid = qid_list[job_idx]
        query = queries[str(qid)]
        q_term_length = len(tokenizer.tokenize(query))
        data_path1 = os.path.join(output_path, "robust", "windowed_{}.score".format(job_idx))
        data_path2 = os.path.join(output_path, "robust", "windowed_small_{}.score".format(job_idx))
        data1 = OutputViewer(data_path1, n_factor, batch_size)
        data2 = OutputViewer(data_path2, n_factor, batch_size)
        segment_len = max_seq_length - 3 - q_term_length
        segment_len2 = max_seq_length2 - 3 - q_term_length

        outputs = []
        for d1, d2 in zip(data1, data2):
            # for each query, doc pairs
            cur_info1 = info[d1['data_id']]
            cur_info2 = info[d2['data_id']]
            query_doc_id1 = f_handler.get_pair_id(cur_info1)
            query_doc_id2 = f_handler.get_pair_id(cur_info2)

            assert query_doc_id1 == query_doc_id2

            doc = d1['doc']
            probs = get_probs(d1['logits'])
            probs2 = get_probs(d2['logits'])
            n_pred_true = np.count_nonzero(np.less(0.5, probs))
            print(n_pred_true, len(probs))

            seg_scores: List[Tuple[int, int, float]] = get_piece_scores(n_factor, probs, segment_len, step_size)
            seg_scores2: List[Tuple[int, int, float]] = get_piece_scores(n_factor, probs2, segment_len2, step_size)
            ss_list = []
            for st, ed, score in seg_scores:
                try:
                    st2, ed2, score2 = find_where(lambda x: x[1] == ed, seg_scores2)
                    assert ed == ed2
                    assert st < st2
                    tokens = tokenizer.convert_ids_to_tokens(doc[st: st2])
                    diff = score - score2
                    ss = ScoredPiece(st, st2, diff, tokens)
                    ss_list.append(ss)
                except StopIteration:
                    pass
            outputs.append((probs, probs2, query_doc_id1, ss_list))

        html = HtmlVisualizer("windowed.html")

        for probs, probs2, query_doc_id, ss_list in outputs:
            html.write_paragraph(str(query_doc_id))
            html.write_paragraph("Query: " + query)

            ss_list.sort(key=lambda ss: ss.st)
            prev_end = None
            cells = []
            prob_str1 = lmap(two_digit_float, probs)
            prob_str1 = ["8.88"] + prob_str1
            prob_str2 = lmap(two_digit_float, probs2)
            html.write_paragraph(" ".join(prob_str1))
            html.write_paragraph(" ".join(prob_str2))

            for ss in ss_list:
                if prev_end is not None:
                    assert prev_end == ss.st
                else:
                    print(ss.st)

                score = abs(int(100 * ss.score))
                color = "B" if score > 0 else "R"
                cells.extend([Cell(t, score, target_color=color) for t in ss.tokens])
                prev_end = ss.ed

            html.multirow_print(cells)


if __name__ == "__main__":
    main()