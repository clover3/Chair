import os
import subprocess
from typing import List, Dict, Set

from base_type import FilePath
from cpath import data_path
from datastore.alchemy_schema import Base, KeyOnlyTable, engine, Session, TokenizedCluewebDocTable
from datastore.interface import has_key, load, save, flush
from datastore.table_names import TokenizedCluewebDoc, RawCluewebDoc, CluewebDocTF, QueryResult
from galagos.parse import load_galago_ranked_list
from galagos.types import GalagoDocRankEntry, QueryResultID
from misc_lib import TimeEstimator
from sydney_clueweb.clue_path import get_first_disk


def load_from_db_or_from_galago(table_name, key, galago_fn):
    if has_key(table_name, key):
        return load(table_name, key)

    r = galago_fn()
    if not has_key(table_name, key):
        save(table_name, key, r)
        flush()
    return r


def insert_ranked_list(q_res_id: QueryResultID,
                       ranked_list: List[GalagoDocRankEntry]):
    save(QueryResult, q_res_id, ranked_list)
    flush()


class DocGetter:
    def __init__(self):
        print("DocGetter __init__")
        self.disk_path = get_first_disk()

    def get_tokened_doc(self, doc_id) -> List[str]:
        return self.get_db_item_or_make(TokenizedCluewebDoc, doc_id)

    def get_raw_doc(self, doc_id) -> str:
        return self.get_db_item_or_make(RawCluewebDoc, doc_id)

    def get_doc_tf(self, doc_id):
        return self.get_db_item_or_make(CluewebDocTF, doc_id)

    def get_db_item_or_make(self, table_name, doc_id):
        if has_key(table_name, doc_id):
            return load(table_name, doc_id)
        print("doc_id not found:",  doc_id)
        self.launch_doc_processor(doc_id)
        return load(table_name, doc_id)

    def launch_doc_processor(self, doc_id):
        env_path = "/mnt/nfs/work3/youngwookim/code/Chair/src"
        voca_path = os.path.join(data_path, "bert_voca.txt")
        cmd = ["/mnt/nfs/work3/youngwookim/miniconda3/envs/boilerplate3/bin/python",
               "src/galagos/runner/get_doc_and_process.py",
               voca_path,
               self.disk_path,
               doc_id,
               ]
        env = os.environ.copy()
        env['PYTHONPATH'] = env_path

        p = subprocess.Popen(cmd,
                             env=env,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             )

        print(p.communicate())


def insert_ranked_list_from_path(file_path: FilePath, q_config_id: str):
    ranked_list: Dict[str, List[GalagoDocRankEntry]] = load_galago_ranked_list(file_path)

    for query_id in ranked_list:
        q_res_id: QueryResultID = QueryResultID("{}_{}".format(query_id, q_config_id))
        insert_ranked_list(q_res_id, ranked_list[query_id])


def add_doc_list_to_table(doc_list, save_name):
    class DocIdTable(Base, KeyOnlyTable):
        __tablename__ = save_name
        extend_existing = True
    Base.metadata.create_all(engine)
    print("Writing doc list to table : ", len(doc_list))
    ticker = TimeEstimator(len(doc_list))

    session = Session()
    cnt = 0

    payload = []
    for key in doc_list:
        ticker.tick()
        new_record = DocIdTable(key=key)
        payload.append(new_record)
        cnt += 1
    session.bulk_save_objects(payload)
    session.flush()
    session.commit()


import logging
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)


def get_docs_in_db(save_name) -> Set:
    class DocIdTable(Base, KeyOnlyTable):
        __tablename__ = save_name

    session = Session()
    print("execute join")
    j = session.query(DocIdTable.key).join(TokenizedCluewebDocTable,
                                           DocIdTable.key == TokenizedCluewebDocTable.key)
    print("issued now getting")

    doc_id_in_db = set()
    for entry in j.all():
        doc_id_in_db.add(entry.key)
    return doc_id_in_db