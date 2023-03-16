import xmlrpc.client

from adhoc.bm25_class import BM25
from cpath import at_output_dir
from dataset_specific.msmarco.passage.load_term_stats import load_msmarco_passage_term_stat
from dataset_specific.msmarco.passage.passage_resource_loader import load_msmarco_sample_a_as_pairs, \
    load_msmarco_sample_dev_as_pairs
from misc_lib import write_to_lines, TELI


def get_bm25() -> BM25:
    cdf, df = load_msmarco_passage_term_stat()
    bm25 = BM25(df, cdf, 25)
    return bm25


def main():
    run_name = "bm25"
    dataset = "dev_sample_a"

    itr = iter(load_msmarco_sample_a_as_pairs())
    save_path = at_output_dir("lines_scores", f"{run_name}_{dataset}.txt")
    f = open(save_path, "w")
    bm25 = get_bm25()
    for q, d in TELI(itr, 1000*1000):
        score = bm25.score(q, d)
        f.write("{}\n".format(score))


def main():
    run_name = "bm25"
    dataset = "dev_sample"
    print(run_name, dataset)

    itr = iter(load_msmarco_sample_dev_as_pairs())
    save_path = at_output_dir("lines_scores", f"{run_name}_{dataset}.txt")
    f = open(save_path, "w")
    bm25 = get_bm25()
    for q, d in TELI(itr, 100*100):
        score = bm25.score(q, d)
        f.write("{}\n".format(score))


if __name__ == "__main__":
    main()