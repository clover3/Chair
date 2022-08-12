import json

from alignment import RelatedEvalAnswer
from alignment.data_structure.matrix_scorer_if import ContributionSummary
from alignment.ists_eval.path_helper import get_ists_2d_save_path


def parse_j(j):
    for problem in j:
        problem_id = problem[0]
        contribution_summary = problem[1]
        assert len(contribution_summary) == 1
        table = contribution_summary[0]
        print("problem_id", problem_id)
        RelatedEvalAnswer(problem, ContributionSummary())




def main():
    run_name = "nlits_mini"
    genre = "headlines"
    split = "train"
    save_path = get_ists_2d_save_path(genre, split, run_name)
    j = json.load(open(save_path, "r"))
    print(j)
    parse_j(j)


if __name__ == "__main__":
    main()