from dataset_specific.scientsbank.parse_fns import sci_ents_test_split_list, get_split_spec
from dataset_specific.scientsbank.save_load_pred import load_pte_preds_from_file, save_pte_preds_to_file
from trainer_v2.per_project.tli.pte.path_helper import get_score_save_path, get_threshold_save_path


def load_threshold(name):
    v = open(get_threshold_save_path(name), "r").read()
    return float(v)


def main():
    solver_name_list = [
        "em", "w2v", "coattention", "lime", "deletion",
        "senli", "nli14", "nlits"]
    solver_name_list = ["slr"]
    split_todo = sci_ents_test_split_list
    print(solver_name_list)
    for solver_name in solver_name_list:
        threshold = load_threshold(solver_name)
        for split_name in split_todo:
            split = get_split_spec(split_name)
            run_name = f"{solver_name}_{split.get_save_name()}"
            try:
                raw_score_path = get_score_save_path(run_name)
                preds = load_pte_preds_from_file(raw_score_path)
                for p in preds:
                    for sa in p.per_student_answer_list:
                        for fp in sa.facet_pred:
                            fp.pred = fp.score > threshold

                new_score_path = get_score_save_path(run_name + "_t")
                save_pte_preds_to_file(preds, new_score_path)
            except FileNotFoundError as e:
                print(e)


if __name__ == "__main__":
    main()