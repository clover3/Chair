import numpy as np
from cpath import common_model_dir_root
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import right, left
from misc_lib import path_join, two_digit_float
from tab_print import print_table
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.pep.emb_train.attn_query_key import QueryKeyValue, \
    get_predictor


def compute_pairwise_sim(dt_list, qt_list, rep_d, similarity_on_tokens):
    head = [""] + dt_list
    table = [head]
    for qt in qt_list:
        row = [qt]
        for dt in dt_list:
            sim = similarity_on_tokens(rep_d[qt], rep_d[dt])
            row.append(two_digit_float(sim))
        table.append(row)
    return table


def main():
    c_log.info(__file__)

    tokenizer = get_tokenizer()
    model_path = path_join(common_model_dir_root, "runs", "mmp_pep10_point", "model_20000")
    predictor = get_predictor(model_path)
    pair_list = [
        ("3rd", "third"),
        ("phone", "53"),
        ("lobbying", "lobbyist")
    ]

    rep_d: dict[str, list[QueryKeyValue]] = {}

    def compute(text, is_query):
        tokens = tokenizer.tokenize(text)
        ret = predictor(tokens, is_query)
        return ret

    for qt, dt in pair_list:
        rep_d[qt] = compute(qt, True)
        rep_d[dt] = compute(dt, False)

    reduce_over_tokens = np.mean
    reduce_over_head_layers = np.mean

    def similarity_per_token(q_token: QueryKeyValue, d_token: QueryKeyValue) -> float:
        sim_arr1 = np.sum(q_token.query * d_token.key, axis=2)  # [Num head, num_layer, H]
        sim_arr2 = np.sum(q_token.key * d_token.query, axis=2)  # [Num head, num_layer, H]
        sim_arr = np.concatenate([sim_arr1, sim_arr2], axis=0)
        sim_arr = sim_arr[:, :4,]
        return reduce_over_head_layers(sim_arr)


    def similarity_on_tokens(q_tokens: list[QueryKeyValue], d_tokens: list[QueryKeyValue]):
        table = []
        for q_token in q_tokens:
            row = []
            for d_token in d_tokens:
                s: float = similarity_per_token(q_token, d_token)
                row.append(s)
            table.append(row)
        return reduce_over_tokens(table)

    qt_list = left(pair_list)
    dt_list = right(pair_list)

    def print_num_tokens(term):
        tokens = tokenizer.tokenize(term)
        print(f"{term} has {len(tokens)} tokens")

    reduce_over_tokens = np.max

    for qt, dt in pair_list:
        print_num_tokens(qt)
        print_num_tokens(dt)

    table = compute_pairwise_sim(dt_list, qt_list, rep_d, similarity_on_tokens)
    print_table(table)

    reduce_over_tokens = np.max
    table = compute_pairwise_sim(dt_list, qt_list, rep_d, similarity_on_tokens)
    print_table(table)

    reduce_over_head_layers = np.max
    table = compute_pairwise_sim(dt_list, qt_list, rep_d, similarity_on_tokens)
    print_table(table)


if __name__ == "__main__":
    main()