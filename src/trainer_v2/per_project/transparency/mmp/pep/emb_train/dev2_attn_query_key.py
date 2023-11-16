import numpy as np
import numpy as np
from cpath import common_model_dir_root
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import right, left
from misc_lib import path_join, two_digit_float
from tab_print import print_table
from table_lib import tsv_iter
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.pep.emb_train.attn_query_key import QueryKeyValue, \
    get_predictor
import heapq


def update_top_k(heap, item: tuple[float, str], k):
    if len(heap) < k:
        heapq.heappush(heap, item)
        # print(f"Added to top-{k} for {d_term}: {token} ({item[0]:.2f})")
        return True
    elif item[0] > heap[0][0]:
        # Replace the smallest element in the heap if the new item is larger
        heapq.heappushpop(heap, item)
        # print(f"Updated top-{k} for {d_term}: {token} ({item[0]:.2f})")
        return True
    return False

#
# def read_voca():
#     def parse_string_pair(text):
#         # Assuming the input is in the format ('string1', 'string2')
#         # Strip the outer parentheses
#         stripped_text = text.strip("()")
#
#         # Split the string by comma and strip extra whitespace and quotes
#         string_pair = [s.strip(" '\"") for s in stripped_text.split(',')]
#
#         return string_pair
#
#     for row in tsv_iter("output/mmp/mmp10_pair_0_non_em.txt"):
#         text1, text2 = parse_string_pair(row[0])


def main():
    c_log.info(__file__)

    tokenizer = get_tokenizer()
    model_path = path_join(common_model_dir_root, "runs", "mmp_pep10_point", "model_20000")
    predictor = get_predictor(model_path)
    pair_list = [
        ("3rd", "third"),
        ("phone number", "539"),
        ("cumple meaning", "cumpleanos")
    ]

    rep_d: dict[str, QueryKeyValue] = {}

    def compute(text, is_query):
        tokens = tokenizer.tokenize(text)
        ret = predictor(tokens, is_query)
        return ret

    reduce_over_tokens = np.mean
    reduce_over_head_layers = np.mean

    d_terms = [
        "3rd", "third", "phone", "lobbying", "usd", "gossip"
    ]
    for dt in d_terms:
        reps = compute(dt, True)
        if not len(reps) == 1:
            print(f"{dt} has more than 1 tokens")
        rep_d[dt] = reps[0]

    per_term_top_k = {d_term: [] for d_term in d_terms}
    k = 3  # Replace with your desired value of k

    for voca_id in range(1997, 20000):
        cand_tokens = tokenizer.convert_ids_to_tokens([voca_id])
        cand_token = cand_tokens[0]
        cand_rep = predictor(cand_tokens, False)[0]
        for d_term in d_terms:
            target_rep = rep_d[d_term]
            sim_arr1 = np.sum(target_rep.query * cand_rep.query, axis=2)  # [Num head, num_layer, H]
            sim_arr2 = np.sum(target_rep.key * cand_rep.key, axis=2)  # [Num head, num_layer, H]
            score = np.mean(sim_arr1) + np.mean(sim_arr2)
            f_change = update_top_k(per_term_top_k[d_term], (score, cand_token), k)
            if f_change:
                heap_desc = ", ".join([f"[{key}]({score: .2f})" for score, key in per_term_top_k[d_term]])
                print(f"[{d_term}] is updated with {cand_token}: {heap_desc}")


if __name__ == "__main__":
    main()
