import time
from collections import Counter
from typing import Callable, List, Any, TypeVar, Dict

FUNC_SIG_GENERIC = Callable[[List[Any]], List[Any]]
InputType = TypeVar('InputType')
OutType = TypeVar('OutType')
HashType = TypeVar('HashType')


class OverheadRecorder:
    def __init__(self):
        self.overhead = 0
        self.st = 0

    def init(self):
        self.overhead = 0

    def start(self):
        self.overhead = 0
        self._resume()

    def end(self, overhead_limit: float = None):
        self._pause()
        overhead = self.overhead
        if overhead_limit is not None and overhead > overhead_limit:
            print("WARNING overhead exceeds limit")

    def pause(self):
        self._pause()

    def resume(self):
        self._resume()

    def _pause(self):
        elapsed = time.time() - self.st
        self.overhead += elapsed

    def _resume(self):
        self.st = time.time()

    def get_last_overhead(self):
        return self.overhead


class MemoryCachedClient:
    def __init__(self,
                 forward_fn: FUNC_SIG_GENERIC,
                 hash_fn: Callable[[InputType], HashType],
                 dictionary: Dict[HashType, OutType],
                 expected_overhead_fn: Callable[[List[InputType]], float] = None
                 ):
        self.forward_fn = forward_fn
        self.dictionary: Dict[HashType, OutType] = dictionary
        self.new_item_dict: Dict[HashType, OutType] = {}
        self.hash_fn: Callable[[InputType], HashType] = hash_fn
        self.hash_overhead = OverheadRecorder()
        self.expected_overhead_fn: Callable[[List[InputType]], float] = expected_overhead_fn
        self.hash_hit_rate = Counter()

    def predict(self, items: List[InputType]) -> List[OutType]:
        self.hash_overhead.start()
        keys: List[HashType] = [self.hash_fn(s) for s in items]
        output_d, todo = self.check_cache(keys)
        self.hash_hit_rate["miss"] += len(todo)
        self.hash_hit_rate["hit"] += len(items) - len(todo)
        self.hash_overhead.pause()
        segs_to_predict: List[InputType] = [items[i] for i in todo]
        new_predictions: List[OutType] = self.forward_fn(segs_to_predict)
        for n_idx, s in enumerate(new_predictions):
            ori_idx = todo[n_idx]
            output_d[ori_idx] = s

        self.hash_overhead.resume()
        self.update_cache(keys, new_predictions, todo)

        overhead_limit: float = self.expected_overhead_fn(items)
        self.hash_overhead.end(overhead_limit)

        output_predictions: List[OutType] = []
        for idx in range(len(items)):
            output_predictions.append(output_d[idx])
        return output_predictions

    def update_cache(self, keys, new_predictions, todo):
        for n_idx, s in enumerate(new_predictions):
            ori_idx = todo[n_idx]
            key = keys[ori_idx]
            self.dictionary[key] = s
            self.new_item_dict[key] = s

    def check_cache(self, keys):
        output_d: Dict[int, OutType] = {}
        todo = []
        for idx, k in enumerate(keys):
            if k in self.dictionary:
                output_d[idx] = self.dictionary[k]
            else:
                todo.append(idx)
        return output_d, todo

    def get_last_overhead(self):
        return self.hash_overhead.get_last_overhead()

    def get_new_items(self):
        return self.new_item_dict

    def reset_new_items(self):
        self.new_item_dict = {}
