from abc import ABC, abstractmethod
from typing import List, Tuple, TypeVar, NamedTuple

from contradiction.medical_claims.token_tagging.problem_loader import AlamriProblem
from contradiction.medical_claims.token_tagging.query_id_helper import get_query_id
from contradiction.medical_claims.token_tagging.trec_entry_helper import convert_token_scores_to_trec_entries
from list_lib import lmap
from trainer.promise import PromiseKeeper, MyFuture, list_future
from trec.trec_parse import write_trec_ranked_list_entry


class BatchTokenScoringSolverIF(ABC):
    @abstractmethod
    def solve(self, payload: List[Tuple[List[str], List[str]]]) -> List[Tuple[List[float], List[float]]]:
        pass


def make_ranked_list_w_batch_solver(problems: List[AlamriProblem],
                                    run_name, save_path, tag_type, solver: BatchTokenScoringSolverIF):
    all_ranked_list = []
    payload = []
    for p in problems:
        input_per_problem = p.text1.split(), p.text2.split()
        payload.append(input_per_problem)

    batch_output = solver.solve(payload)
    assert len(problems) == len(batch_output)
    for p, output in zip(problems, batch_output):
        def get_query_id_inner(sent_name):
            return get_query_id(p.group_no, p.inner_idx, sent_name, tag_type)

        scores1, scores2 = output
        rl1 = convert_token_scores_to_trec_entries(get_query_id_inner('prem'), run_name, scores1)
        rl2 = convert_token_scores_to_trec_entries(get_query_id_inner('hypo'), run_name, scores2)

        all_ranked_list.extend(rl1)
        all_ranked_list.extend(rl2)
    write_trec_ranked_list_entry(all_ranked_list, save_path)


NeuralInput = TypeVar('NeuralInput')
NeuralOutput = TypeVar('NeuralOutput')
ECCInput = Tuple[List[str], List[str]]
ECCOutput = Tuple[List[float], List[float]]


# Batch Solver Adapter
class BSAdapterIF(ABC):
    @abstractmethod
    def neural_worker(self, items: List):
        pass

    @abstractmethod
    def reduce(self, t1, t2, output: List) -> List[float]:
        pass

    @abstractmethod
    def enum_child(self, t1: List[str], t2: List[str]) -> List:
        pass


class BatchSolver(BatchTokenScoringSolverIF):
    def __init__(self, adapter):
        self.adapter: BSAdapterIF = adapter

    def solve(self, payload: List[ECCInput]) -> List[ECCOutput]:
        pk = PromiseKeeper(self.adapter.neural_worker)

        class Entry(NamedTuple):
            t1: List[str]
            t2: List[str]
            f_list1: List[MyFuture]
            f_list2: List[MyFuture]

        def prepare_pre_problem(input_obj: ECCInput) -> Entry:
            t1, t2 = input_obj
            es_list1: List[NeuralInput] = self.adapter.enum_child(t2, t1)
            es_list2: List[NeuralInput] = self.adapter.enum_child(t1, t2)
            future_list1 = lmap(pk.get_future, es_list1)
            future_list2 = lmap(pk.get_future, es_list2)
            return Entry(t1, t2, future_list1, future_list2)

        future_ref = lmap(prepare_pre_problem, payload)
        pk.do_duty()

        def apply_reduce(e: Entry) -> ECCOutput:
            def get_scores(t1, t2, fl: List[MyFuture[NeuralOutput]]) -> List[float]:
                fl: List[MyFuture[NeuralOutput]] = fl
                l:  List[NeuralOutput] = list_future(fl)
                scores: List[float] = self.adapter.reduce(t1, t2, l)
                return scores
            return get_scores(e.t2, e.t1, e.f_list1), \
                   get_scores(e.t1, e.t2, e.f_list2)

        return lmap(apply_reduce, future_ref)


