from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import get_shallow_score_save_path


def save_shallow_scores(qid_scores, save_path):
    f = open(save_path, "w")
    for qid, items in qid_scores:
        for pid, score in items:
            f.write("\t".join(map(str, [qid, pid, score])) + "\n")