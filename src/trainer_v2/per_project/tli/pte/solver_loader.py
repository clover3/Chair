from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIFOneWay
from data_generator.NLI.enlidef import ENTAILMENT, enli_tags
from dataset_specific.scientsbank.pte_solver_if import PTESolverIF
from trainer_v2.custom_loop.attention_helper.model_shortcut import nlits87_model_path
from trainer_v2.custom_loop.run_config2 import get_run_config_for_predict
from trainer_v2.per_project.tli.enum_subseq import enum_subseq_136
from trainer_v2.per_project.tli.pte.nlits_solver import PTESolverTLI
from trainer_v2.per_project.tli.pte.solver_adapter import PTESolverFromTokenScoringSolver
from trainer_v2.per_project.tli.runner.inference_all_train_split import get_predictor
from trainer_v2.train_util.arg_flags import flags_parser
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def get_deletion_solver(target_idx) -> TokenScoringSolverIFOneWay:
    from trainer_v2.keras_server.name_short_cuts import get_nli14_direct
    from contradiction.medical_claims.token_tagging.solvers.deletion_solver import DeletionSolverKeras

    predict_fn = get_nli14_direct(get_strategy())
    return DeletionSolverKeras(predict_fn, target_idx)


def get_token_scoring_solver_by_name(name) -> TokenScoringSolverIFOneWay:
    target_idx = ENTAILMENT
    if name == "deletion":
        return get_deletion_solver(target_idx)
    elif name == "senli":
        from contradiction.medical_claims.token_tagging.solvers.senli_solver import get_senli_solver
        return get_senli_solver(enli_tags[target_idx])
    elif name == "coattention":
        from contradiction.medical_claims.token_tagging.solvers.coattention_solver import get_co_attention_solver
        return get_co_attention_solver()
    elif name == "lime":
        from contradiction.medical_claims.token_tagging.solvers.lime_solver import get_lime_solver_nli14_direct
        return get_lime_solver_nli14_direct(target_idx)
    else:
        raise KeyError()


def get_nlits_pte_solver(name):
    args = flags_parser.parse_args("")
    run_config = get_run_config_for_predict(args)
    run_config.predict_config.model_save_path = nlits87_model_path()
    nli_predict_fn = get_predictor(run_config)
    combine_tli = lambda x: x[:, 0]
    solver = PTESolverTLI(nli_predict_fn, enum_subseq_136, combine_tli, name)
    return solver


def get_solver_by_name(name) -> PTESolverIF:
    if name == "nlits":
        return get_nlits_pte_solver(name)
    token_solver = get_token_scoring_solver_by_name(name)
    def tokenizer(text):
        return text.lower().split()

    reverse_score = name == "coattention"
    solver = PTESolverFromTokenScoringSolver(token_solver, tokenizer, reverse_score, name)
    return solver
