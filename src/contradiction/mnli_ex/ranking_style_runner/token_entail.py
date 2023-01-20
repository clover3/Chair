

from contradiction.medical_claims.token_tagging.solvers.search_solver import PartialSegSolver, WordSegSolver
from contradiction.mnli_ex.ranking_style_helper import solve_mnli_tag
from data_generator.NLI.enlidef import get_target_class, enli_tags
from trainer_v2.keras_server.name_short_cuts import get_keras_nli_300_client


def do_for_label(tag_type, split):
    predict_fn = get_keras_nli_300_client().request_multiple
    target_idx = get_target_class(tag_type)
    solver = WordSegSolver(target_idx, predict_fn)
    run_name = "token_entail"
    solve_mnli_tag(split, run_name, solver, tag_type)


def main():
    for split in ["dev", "test"]:
        for tag_type in enli_tags:
            do_for_label(tag_type, split)


if __name__ == "__main__":
    main()
