from data_generator.tokenizer_wo_tf import get_tokenizer
from cpath import output_path
from misc_lib import path_join
from table_lib import tsv_iter


def get_1000_3000_terms():
    tokenizer = get_tokenizer()
    all_terms = []
    for i in range(1000, 3000):
        term = tokenizer.inv_vocab[i]
        all_terms.append(term)
    print(all_terms[:10], all_terms[-10:])
    return all_terms


def main():
    terms_1000_3000 = get_1000_3000_terms()
    candidate_set_name = "candidate2_1_new"
    save_path = path_join(
        output_path, "msmarco", "passage", "align_candidates",
        f"{candidate_set_name}.tsv")

    items = list(tsv_iter(save_path))

    for _q_term, d_term in items:
        if d_term not in terms_1000_3000:
            print(d_term, "is not in 1000_3000")


if __name__ == "__main__":
    main()