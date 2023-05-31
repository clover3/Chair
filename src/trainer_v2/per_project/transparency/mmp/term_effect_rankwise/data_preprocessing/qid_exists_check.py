import os.path

from misc_lib import SuccessCounter
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import load_qtfs, \
    get_deep_score_save_path_by_qid
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_mmp_split_w_deep_scores


def main():
    job_no_itr = get_mmp_split_w_deep_scores()
    print(list(job_no_itr))
    # job_no_itr = [46]
    for job_no in job_no_itr:
        suc = SuccessCounter()
        for qid, _ in load_qtfs(job_no):
            check_path = get_deep_score_save_path_by_qid(qid)
            if not os.path.exists(check_path):
                suc.fail()
                # print("Missing qid {}".format(qid))
            else:
                suc.suc()

        print(job_no, suc.get_summary())

if __name__ == "__main__":
    main()