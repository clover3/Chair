from dataset_specific.msmarco.passage.processed_resource_loader import get_queries_path
from list_lib import left, flatten
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_mmp_split_w_deep_scores
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.term_effect_measure_mmp import load_deep_scores
from trec.ranked_list_util import build_ranked_list
from trec.trec_parse import write_trec_ranked_list_entry
from cpath import output_path
from misc_lib import path_join


def do_for_dataset(dataset):
    qid_queries = list(tsv_iter(get_queries_path(dataset)))
    run_name = "mmp1"
    target_qids = set(left(qid_queries))
    print(f"{len(target_qids)} qids")
    output = []
    output_rl = []
    split = "dev"
    for partition in get_mmp_split_w_deep_scores(split):
        print(f"Partition {partition}")
        grouped = load_deep_scores(split, partition)
        for group in grouped:
            qid = group[0][0]
            if qid in target_qids:
                output.append(group)
                print(f"{len(output)} found")
                pid_scores = [(pid, score) for qid, pid, score in group]
                rl = build_ranked_list(qid, run_name, pid_scores)
                output_rl.append(rl)

            if len(output) == len(target_qids):
                break

        if len(output) == len(target_qids):
            break
    save_path = path_join(output_path, "ranked_list", f"{run_name}_{dataset}.txt")
    write_trec_ranked_list_entry(flatten(output_rl), save_path)


def main():
    dataset = "dev_sample100"
    do_for_dataset(dataset)
    dataset = "dev_sample1000"
    do_for_dataset(dataset)


if __name__ == "__main__":
    main()