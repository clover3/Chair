from typing import List, Callable, Generic, TypeVar

from alignment import RelatedEvalInstance
from bert_api.task_clients.nli_interface.nli_interface import NLIInput
from datastore.cached_client import MemoryCachedClient
from misc_lib import tprint


class NonDuplicateJobEnum:
    def __init__(self, hash_fn, forward_fn):
        self.volatile_cache_client: MemoryCachedClient = MemoryCachedClient(
            forward_fn,
            hash_fn,
            {},
            None
        )
        self.g_n_calc = 0

    def predict(self, segs: List[NLIInput]) -> List[List[float]]:
        items = self.volatile_cache_client.predict(segs)
        return items


T = TypeVar('T')
class StoreAndIter(Generic[T]):
    def __init__(self, dummy_val):
        self.items = []
        self.dummy_val = dummy_val

    def forward_fn(self, items: List):
        self.items.extend(items)
        return [self.dummy_val for _ in items]

    def pop_items(self):
        items = self.items
        self.items = []
        return items


class JobPayloadSaver:
    def __init__(self, payload_save_fn: Callable[[int, List], None], n_item_per_job):
        self.n_item_per_job = n_item_per_job
        self.payload_save_fn = payload_save_fn

    def run_with_itr(self, itr):
        cur_job= []
        job_id = 0
        for item in itr:
            cur_job.append(item)
            if len(cur_job) == self.n_item_per_job:
                self.payload_save_fn(job_id, cur_job)
                tprint("job {} written".format(job_id))
                job_id += 1
                cur_job = []
        if cur_job:
            self.payload_save_fn(job_id, cur_job)