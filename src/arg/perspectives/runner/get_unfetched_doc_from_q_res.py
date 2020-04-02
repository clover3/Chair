# Query DB to get ranked list
# build set of doc_id in the all ranked list
# filter doc_id that are not in the db
# save the list of doc_id to file

import os
import sys
from typing import List

from arg.perspectives.basic_analysis import load_train_data_point
from arg.perspectives.dp_query_routines import dp_to_qid
from cache import save_to_pickle
from cpath import output_path
from datastore.alchemy_schema import KeyOnlyTable, Base, Session, RawCluewebDocTable, engine
from datastore.interface import has_key, load
from datastore.table_names import QueryResult
from galagos.query_runs_ids import Q_CONFIG_ID_BM25_10000
from galagos.types import GalagoDocRankEntry
from list_lib import foreach, lmap
from misc_lib import exist_or_mkdir, TimeEstimator


def read_doc_list(st, ed):
    st = int(st)
    ed = int(ed)
    q_config_id = Q_CONFIG_ID_BM25_10000
    all_data_points = load_train_data_point()

    print("Running {}~{} of {}".format(st, ed, len(all_data_points)))

    todo = all_data_points[st:ed]
    qid_list = lmap(dp_to_qid, todo)

    doc_list = set()

    ticker = TimeEstimator(len(qid_list))
    def get_doc_list(query_id: str):
        q_res_id: str = "{}_{}".format(query_id, q_config_id)
        ticker.tick()
        if has_key(QueryResult, q_res_id):
            r: List[GalagoDocRankEntry] = load(QueryResult, q_res_id)

            for entry in r:
                doc_id, rank, score = entry
                doc_list.add(doc_id)

    print("parsing_doc_list")
    foreach(get_doc_list, qid_list)

    return doc_list


def add_doc_list_to_table(doc_list, save_name):
    class DocIdTable(Base, KeyOnlyTable):
        __tablename__ = save_name

    Base.metadata.create_all(engine)
    print("Writing doc list to table : ", len(doc_list))
    ticker = TimeEstimator(len(doc_list))

    session = Session()
    cnt = 0
    def add_to_table(key):
        ticker.tick()
        new_record = DocIdTable(key=key)
        session.add(new_record)
        nonlocal cnt
        cnt += 1
        if cnt > 1000:
            session.flush()
            cnt = 0

    foreach(add_to_table, doc_list)
    session.flush()
    session.commit()


def do_join_and_write(doc_list, save_name):
    class DocIdTable(Base, KeyOnlyTable):
        __tablename__ = save_name

    session = Session()
    print("execute join")
    j = session.query(DocIdTable).join(RawCluewebDocTable, DocIdTable.key == RawCluewebDocTable.key)

    doc_id_in_db = set()
    for entry in j.all():
        doc_id_in_db.add(entry)

    doc_list_to_fetch = doc_list - doc_id_in_db
    exist_or_mkdir(os.path.join(output_path, "doc_list"))
    save_path = os.path.join(output_path, "doc_list", save_name)
    f = open(save_path, "w")
    write = lambda doc_id: f.write("{}\n".format(doc_id))
    foreach(write, doc_list_to_fetch)
    f.close()


def work(st, ed, save_name):
    doc_list = read_doc_list(st, ed)
    run_name = "{}_{}_{}".format(st, ed, save_name)
    save_to_pickle(doc_list, run_name)
    #add_doc_list_to_table(doc_list, save_name)
    do_join_and_write(doc_list, save_name)


if __name__ == "__main__":
    work(sys.argv[1], sys.argv[2], sys.argv[3])
