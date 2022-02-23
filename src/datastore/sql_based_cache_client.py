from typing import List, Callable, Dict, TypeVar, Generic

import sqlalchemy.exc

from datastore.cached_client import MemoryCachedClient
from tlm.qtype.partial_relevance.cache_db import bulk_save, read_cache_from_sqlite, build_db, \
    read_cache_s_from_sqlite, build_db_s, bulk_save_s

T = TypeVar('T')
V = TypeVar('V')


class SQLBasedCacheClient(Generic[T, V]):
    def __init__(self,
                 forward_fn: Callable[[List[T]], List[V]],
                 hash_fn: Callable[[T], str],
                 overhead_per_item=None,
                 save_path=None,
                 save_interval=100):
        try:
            dictionary: Dict[str, V] = read_cache_from_sqlite(save_path)
        except sqlalchemy.exc.OperationalError:
            print("Initializing Database")
            build_db(save_path)
            dictionary: Dict[str, V] = {}

        def overhead_calc(items: List[T]) -> float:
            if overhead_per_item is None:
                return len(items) * (0.035 * 2)
            else:
                return len(items) * overhead_per_item

        self.volatile_cache_client: MemoryCachedClient = MemoryCachedClient(
            forward_fn,
            hash_fn,
            dictionary,
            overhead_calc
        )
        self.save_path = save_path
        self.save_per_prediction = save_interval

    def predict(self, segs: List[T]) -> List[V]:
        ret: List[V] = self.volatile_cache_client.predict(segs)
        n_new = len(self.volatile_cache_client.get_new_items())
        if self.save_per_prediction <= n_new:
            self.save_cache()
        return ret

    def save_cache(self):
        bulk_save(self.save_path, self.volatile_cache_client.get_new_items())
        self.volatile_cache_client.reset_new_items()

    def get_last_overhead(self):
        return self.volatile_cache_client.get_last_overhead()


class SQLBasedCacheClientS(Generic[T, V]):
    def __init__(self,
                 forward_fn: Callable[[List[T]], List[V]],
                 hash_fn: Callable[[T], str],
                 overhead_per_item=None,
                 save_path=None,
                 save_interval=100):
        try:
            dictionary: Dict[str, V] = read_cache_s_from_sqlite(save_path)
        except sqlalchemy.exc.OperationalError:
            print("Initializing Database")
            build_db_s(save_path)
            dictionary: Dict[str, V] = {}

        def overhead_calc(items: List[T]) -> float:
            if overhead_per_item is None:
                return len(items) * (0.035 * 2)
            else:
                return len(items) * overhead_per_item

        self.volatile_cache_client: MemoryCachedClient = MemoryCachedClient(
            forward_fn,
            hash_fn,
            dictionary,
            overhead_calc
        )
        self.save_path = save_path
        self.save_per_prediction = save_interval

    def predict(self, segs: List[T]) -> List[V]:
        ret: List[V] = self.volatile_cache_client.predict(segs)
        n_new = len(self.volatile_cache_client.get_new_items())
        if self.save_per_prediction <= n_new:
            self.save_cache()
        return ret

    def save_cache(self):
        bulk_save_s(self.save_path, self.volatile_cache_client.get_new_items())
        self.volatile_cache_client.reset_new_items()

    def get_last_overhead(self):
        return self.volatile_cache_client.get_last_overhead()