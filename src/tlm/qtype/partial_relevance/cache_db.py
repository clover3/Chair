import time
from typing import Dict

from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.exc import NoResultFound

from cpath import at_output_dir
from datastore.cache_sql import get_engine_from_sqlite_path, CacheTable, Base, index_table
from misc_lib import tprint


def get_cache_sqlite_path():
    return at_output_dir("qtype", "mmd_z_cache.sqlite")


def has_key(session, table_class, key):
    try:
        q_res = session.query(table_class).filter(table_class.key == key).one()
        byte_arr = q_res.value
        return True
    except NoResultFound as e:
        return False


def bulk_save(sqlite_path, key_and_value_list):
    # old_path = sqlite_path + ".old"
    # if os.path.exists(old_path):
    #     os.remove(old_path)
    # os.rename(sqlite_path, sqlite_path + ".old")
    #
    # build_db(sqlite_path)
    tprint("bulk_save ENTRY")
    engine = get_engine_from_sqlite_path(sqlite_path)
    session_maker = sessionmaker(bind=engine)
    session = session_maker()
    for key, value in key_and_value_list.items():
        if not has_key(session, CacheTable, key):
            e = CacheTable(key=key, value=value)
            session.add(e)
            session.flush()
    session.commit()
    tprint("bulk_save EXIT")


def read_cache_from_sqlite(sqlite_path) -> Dict:
    st = time.time()
    engine = get_engine_from_sqlite_path(sqlite_path)
    session_maker = sessionmaker(bind=engine)
    with session_maker() as session:
        q_res_itr = session.query(CacheTable).all()
        d = {}
        for row in q_res_itr:
            d[row.key] = row.value
        ed = time.time()
        print("{0} items read at {1:.2f}sec".format(len(d), ed - st))
    return d


def build_db(sqlite_path):
    engine = get_engine_from_sqlite_path(sqlite_path)
    Base.metadata.create_all(engine)
    index_table(CacheTable, engine)