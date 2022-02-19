from typing import List, Callable

from sqlalchemy.orm import sessionmaker

from cpath import at_output_dir
from datastore.cache_sql import get_engine_from_sqlite_path, CacheTableF
from datastore.sql_based_cache_client import SQLBasedCacheClient
from tlm.qtype.partial_relevance.cache_db import get_cache_sqlite_path
from tlm.qtype.partial_relevance.eval_data_structure import SegmentedInstance, get_test_segment_instance
from tlm.qtype.partial_relevance.runner.run_eval_old.run_partial_related_full_eval import get_mmd_client

FUNC_SIG = Callable[[List[SegmentedInstance]], List[float]]


def get_mmd_cache_client(option, hooking_fn=None) -> SQLBasedCacheClient:
    forward_fn_raw: Callable[[List[SegmentedInstance]], List[float]] = get_mmd_client(option)
    if hooking_fn is not None:
        def forward_fn(items: List[SegmentedInstance]) -> List[float]:
            hooking_fn(items)
            return forward_fn_raw(items)
    else:
        forward_fn = forward_fn_raw

    cache_client = SQLBasedCacheClient(forward_fn,
                                       SegmentedInstance.str_hash,
                                       0.035,
                                       get_cache_sqlite_path())
    return cache_client


def test_save():
    segment_instance: SegmentedInstance = get_test_segment_instance()
    forward_fn = get_mmd_client("localhost")
    save_path = get_cache_sqlite_path() + ".test"
    cache_client = SQLBasedCacheClient(forward_fn, save_path)
    result = cache_client.predict([segment_instance])
    cache_client.save_cache()
    print(cache_client.volatile_cache_client.hash_hit_rate)


def db_test():
    sqlite_path = at_output_dir("qtype", "temp.db")
    # build_db(sqlite_path)
    engine = get_engine_from_sqlite_path(sqlite_path)
    key = "something"
    value = 0.42
    session_maker = sessionmaker(bind=engine)
    session = session_maker()
    e = CacheTableF(key=key, value=value)
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
