from dataset_specific.msmarco.passage.passage_resource_loader import load_msmarco_sub_samples_as_qd_pair
from trainer_v2.per_project.transparency.mmp.bm25_paramed import get_bm25_mmp_25_01_01
from trainer_v2.per_project.transparency.mmp.eval_helper.eval_line_format import predict_and_save_scores_w_itr, eval_on_train_when_0


def main():
    run_name = "bm25"
    bm25 = get_bm25_mmp_25_01_01()
    dataset = "train_when_0"
    n_item = 230958
    itr = load_msmarco_sub_samples_as_qd_pair(dataset)
    predict_and_save_scores_w_itr(bm25.score, dataset, run_name, itr, n_item)
    score = eval_on_train_when_0(run_name)
    print(f"MRR:\t{score}")


if __name__ == "__main__":
    main()