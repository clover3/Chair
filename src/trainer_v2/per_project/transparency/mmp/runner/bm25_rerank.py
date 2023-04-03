from adhoc.bm25_class import BM25
from cpath import at_output_dir
from dataset_specific.msmarco.passage.load_term_stats import load_msmarco_passage_term_stat
from dataset_specific.msmarco.passage.passage_resource_loader import load_msmarco_sample_a_as_pairs, \
    load_msmarco_sample_dev_as_pairs
from misc_lib import write_to_lines, TELI
from trainer_v2.per_project.transparency.mmp.eval_helper.eval_line_format import predict_and_save_scores, \
    eval_dev100_for_tune


def get_bm25() -> BM25:
    cdf, df = load_msmarco_passage_term_stat()
    bm25 = BM25(df, cdf, 25)
    return bm25


def main_1000():
    run_name = "bm25_t1"
    dataset = "dev_sample1000"
    cdf, df = load_msmarco_passage_term_stat()
    bm25 = BM25(df, cdf, 25, 0.1, 100, 1.4)
    predict_and_save_scores(bm25.score, dataset, run_name, 1000*1000)
    score = eval_dev100_for_tune(dataset, run_name)
    print(f"ndcg:\t{score}")


def main():
    run_name = "bm25"
    dataset = "dev_sample100"
    print(run_name, dataset)
    bm25 = get_bm25()


    itr = iter(load_msmarco_sample_dev_as_pairs())
    save_path = at_output_dir("lines_scores", f"{run_name}_{dataset}.txt")
    f = open(save_path, "w")
    for q, d in TELI(itr, 100*100):
        score = bm25.score(q, d)
        f.write("{}\n".format(score))


if __name__ == "__main__":
    main_1000()