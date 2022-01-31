from typing import List, Callable, Dict

import sqlalchemy.exc
from sqlalchemy.orm import sessionmaker

from cpath import at_output_dir
from datastore.cache_sql import get_engine_from_sqlite_path, CacheTable
from datastore.cached_client import CachedClient
from tlm.qtype.partial_relevance.cache_db import get_cache_sqlite_path, bulk_save, read_cache_from_sqlite, build_db
from tlm.qtype.partial_relevance.eval_data_structure import SegmentedInstance, get_test_segment_instance
from tlm.qtype.partial_relevance.runner.run_eval.run_partial_related_full_eval import get_mmd_client

FUNC_SIG = Callable[[List[SegmentedInstance]], List[float]]


class MMDCacheClient:
    def __init__(self, forward_fn, save_path=None, save_per_prediction=False):
        if save_path is None:
            save_path = get_cache_sqlite_path()
        try:
            dictionary: Dict[str, float] = read_cache_from_sqlite(save_path)
        except sqlalchemy.exc.OperationalError:
            print("Initializing Database")
            build_db(save_path)
            dictionary: Dict[str, float] = {}

        def overhead_calc(items: List[SegmentedInstance]) -> float:
            return len(items) * (0.035 * 2)

        self.inner_cache_client: CachedClient = CachedClient(
            forward_fn,
            SegmentedInstance.str_hash,
            dictionary,
            overhead_calc
        )
        self.save_path = save_path
        self.save_per_prediction = save_per_prediction

    def predict(self, segs: List[SegmentedInstance]) -> List[float]:
        ret = self.inner_cache_client.predict(segs)
        if self.save_per_prediction:
            self.save_cache()
        return ret

    def save_cache(self):
        bulk_save(self.save_path, self.inner_cache_client.get_new_items())
        self.inner_cache_client.reset_new_items()

    def get_last_overhead(self):
        return self.inner_cache_client.get_last_overhead()


def get_mmd_cache_client(option) -> MMDCacheClient:
    """

    :rtype: object
    """
    forward_fn = get_mmd_client(option)
    cache_client = MMDCacheClient(forward_fn, get_cache_sqlite_path())
    return cache_client


def test_save():
    segment_instance: SegmentedInstance = get_test_segment_instance()
    forward_fn = get_mmd_client("localhost")
    save_path = get_cache_sqlite_path() + ".test"
    cache_client = MMDCacheClient(forward_fn, save_path)
    result = cache_client.predict([segment_instance])
    cache_client.save_cache()
    print(cache_client.inner_cache_client.hash_hit_rate)


def db_test():
    sqlite_path = at_output_dir("qtype", "temp.db")
    # build_db(sqlite_path)
    engine = get_engine_from_sqlite_path(sqlite_path)
    key = "something"
    value = 0.42
    session_maker = sessionmaker(bind=engine)
    session = session_maker()
    e = CacheTable(key=key, value=value)
    session.add(e)
    session.flush()
    session.commit()


def get_engine():
    engine = get_engine_from_sqlite_path(get_cache_sqlite_path())
    return engine


def main():
    # Base.metadata.create_all(get_engine())
    # index_table(CacheTable, get_engine_from_sqlite_path(get_cache_sqlite_path()))
    # d = load_cache(get_cache_path())
    test_save()


if __name__ == "__main__":
    main()
