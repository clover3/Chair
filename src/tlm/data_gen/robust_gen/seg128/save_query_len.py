from cpath import at_output_dir
from data_generator.data_parser.robust import load_robust_04_query
from data_generator.tokenizer_wo_tf import get_tokenizer
from tlm.robust.load import get_robust_qid_list


def main():
    query_type = "desc"
    queries = load_robust_04_query(query_type)
    qid_list = get_robust_qid_list()
    tokenizer = get_tokenizer()

    f = open(at_output_dir("robust", "desc_query_len.txt"), "w")
    for qid in qid_list:
        query = queries[str(qid)]
        query_tokens = tokenizer.tokenize(query)
        n_terms = len(query_tokens)
        f.write("{}\n".format(n_terms))
    f.close()


if __name__ == "__main__":
    main()