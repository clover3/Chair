import os
import subprocess
from typing import List

from cpath import data_path
from datastore.interface import has_key, load, save, flush
from datastore.table_names import TokenizedCluewebDoc, RawCluewebDoc, CluewebDocTF
from sydney_clueweb.clue_path import get_first_disk


def load_from_db_or_from_galago(table_name, key, galago_fn):
    if has_key(table_name, key):
        return load(table_name, key)

    r = galago_fn()
    if not has_key(table_name, key):
        save(table_name, key, r)
        flush()
    return r


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