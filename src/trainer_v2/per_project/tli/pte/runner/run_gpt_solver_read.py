import logging
import sys
import time
2
import openai
from openai.error import RateLimitError, ServiceUnavailableError, OpenAIError

from taskman_client.wrapper3 import JobContext
from utils.open_ai_api import ENGINE_GPT4, ENGINE_GPT_3_5
from dataset_specific.scientsbank.eval_helper import solve_eval_report
from dataset_specific.scientsbank.parse_fns import get_split_spec, load_scientsbank_split
from dataset_specific.scientsbank.pte_data_types import PTEPredictionPerQuestion, Question
from dataset_specific.scientsbank.pte_solver_if import apply_solver
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.tli.pte.gpt_solver import get_gpt_requester, get_gpt_read_solver
from typing import List, Iterable, Callable, Dict, Tuple, Set


def solve_for_split(split_name):
    c_log.setLevel(logging.DEBUG)
    split = get_split_spec(split_name)
    engine = ENGINE_GPT_3_5
    solver = get_gpt_read_solver(engine, split_name)
    #
    solve_eval_report(solver, split)


def main():
    split_name = sys.argv[1]
    solve_for_split(split_name)


if __name__ == "__main__":
    main()