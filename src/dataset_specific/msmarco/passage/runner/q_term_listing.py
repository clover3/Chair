
# Enum queries of dev 1000
from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.msmarco.passage.processed_resource_loader import get_queries_path
from list_lib import right
from table_lib import tsv_iter
from cpath import output_path, data_path
from misc_lib import path_join



def main():
    tokenizer = get_tokenizer()
    tokenize_fn = tokenizer.basic_tokenizer.tokenize

    dataset = "dev_sample1000"
    queries = right(tsv_iter(get_queries_path(dataset)))

    q_terms_set = set()
    for query in queries:
        terms = tokenize_fn(query)
        q_terms_set.update(terms)

    print(f"{dataset} has {len(q_terms_set)} terms.")

    freq_q_term_path = path_join(output_path, "msmarco/passage/freq_q_terms.txt")
    freq_q_terms = [line.strip() for line in open(freq_q_term_path, "r")]

    n_common = len(q_terms_set.intersection(freq_q_terms))
    print(f"It matches {n_common} of {len(freq_q_terms)} frequent q_terms")
    save_path = path_join(data_path, "msmarco", dataset, "query_terms.txt")
    print("Saved at ", save_path)
    with open(save_path, "w") as f:
        for term in q_terms_set:
            f.write(term + "\n")


if __name__ == "__main__":
    main()