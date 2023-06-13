# helper indicates that this functions may not be essential and used to handle some dirty codings


from dataset_specific.scientsbank.pte_solver_if import PTESolverIF, apply_solver
from typing import List, Iterable, Callable, Dict, Tuple, Set

from dataset_specific.scientsbank.eval_fns import solve_and_eval
from dataset_specific.scientsbank.parse_fns import load_scientsbank_split, SplitSpec
from dataset_specific.scientsbank.pte_data_types import PTEPredictionPerQuestion, Question
from dataset_specific.scientsbank.save_load_pred import save_pte_preds_to_file
from misc_lib import ceil_divide
from tab_print import tab_print_dict
from taskman_client.task_proxy import get_task_manager_proxy
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.tli.pte.path_helper import get_score_save_path


def solve_eval_report(solver: PTESolverIF, split: SplitSpec) -> dict:
    questions: List[Question] = load_scientsbank_split(split)
    c_log.info("Predicting on %d questions", len(questions))

    solver_name = solver.get_name()
    if solver_name:
        run_name = f"{solver_name}_{split.get_save_name()}"
        save_path = get_score_save_path(run_name)
    else:
        c_log.info("Run name is not specified. Skip saving")
        save_path = None

    eval_res = solve_and_eval(solver, questions, save_path)
    tab_print_dict(eval_res)
    name = solver.get_name()
    report_macro_f1(eval_res, name, split)
    return eval_res


def solve_part(
        solver: PTESolverIF,
        split: SplitSpec,
        job_no: int):
    all_questions: List[Question] = load_scientsbank_split(split)
    step_size: int = ceil_divide(len(all_questions), 10)
    st: int = job_no * step_size
    ed: int = st + step_size
    questions = all_questions[st:ed]

    c_log.info("Predicting on %d questions", len(questions))

    solver_name = solver.get_name()
    if solver_name:
        run_name = f"{solver_name}_{split.get_save_name()}_{job_no}"
        save_path = get_score_save_path(run_name)
    else:
        c_log.info("Run name is not specified. Skip saving")
        save_path = None

    preds: List[PTEPredictionPerQuestion] = apply_solver(solver, questions)
    save_pte_preds_to_file(preds, save_path)



def report_macro_f1(eval_res, name, split):
    field = 'macro_f1'
    number = eval_res[field]
    proxy = get_task_manager_proxy()
    proxy.report_number(name, number, split.get_save_name(), field)