import json
import os
import sys

from arg.qck.multi_file_save_to_trec_form import multi_file_save_to_trec_form_fn


def make_ranked_list_from_multiple_files(
        run_name,
        save_dir,
        file_idx_start,
        file_idx_end):

    template_json = "data/run_config/robust_trec_save_qck5.json"
    j = json.load(open(template_json, "r"))
    j['run_name'] = run_name
    j['prediction_dir'] = save_dir
    j['file_idx_st'] = file_idx_start
    j['file_idx_ed'] = file_idx_end

    multi_file_save_to_trec_form_fn(save_dir,
                                    file_idx_start,
                                    file_idx_end,
                                    j)


def main():
    model_name = sys.argv[1]
    step = int(sys.argv[2])
    run_name = "{}_{}".format(model_name, step)
    save_root = "/mnt/disks/disk500/robust_score"
    save_dir = os.path.join(save_root, "{}.score".format(run_name))
    print("Loading scores from")
    print(save_dir)
    file_idx_start = 200
    file_idx_end = 250
    make_ranked_list_from_multiple_files(run_name, save_dir, file_idx_start, file_idx_end)


if __name__ == "__main__":
    main()