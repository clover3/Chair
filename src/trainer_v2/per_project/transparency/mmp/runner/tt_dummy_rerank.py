from pytrec_eval import RelevanceEvaluator

from trainer_v2.per_project.transparency.mmp.eval_helper.eval_line_format import predict_and_save_scores, \
    eval_dev100_mrr
from trainer_v2.per_project.transparency.mmp.runner.load_tt_dev import get_dummy_tt_scorer

def main():
    run_name = "tt_dummy"
    dataset = "dev_sample100"
    print(run_name, dataset)
    score_fn = get_dummy_tt_scorer()
    predict_and_save_scores(score_fn, dataset, run_name, 100*100)
    score = eval_dev100_mrr(dataset, run_name)
    print(f"Recip_rank:\t{score}")



if __name__ == "__main__":
    main()