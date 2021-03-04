from typing import Set

from dataset_specific.msmarco.common import per_query_root, get_per_query_doc_path, MSMarcoDataReader, open_top100
from misc_lib import get_second, exist_or_mkdir, tprint, TimeEstimator


def collect_doc_per_query(split):
    ms_reader = MSMarcoDataReader(split)

    def pop(query_id, cur_doc_ids: Set):
        num_candidate_doc = len(cur_doc_ids)
        cur_doc_ids.update(ms_reader.qrel[query_id])
        todo = []
        for doc_id in cur_doc_ids:
            offset = ms_reader.doc_offset[doc_id]
            todo.append((doc_id, offset))
        todo.sort(key=get_second)
        num_all_docs = len(cur_doc_ids)

        exist_or_mkdir(per_query_root)
        save_path = get_per_query_doc_path(query_id)
        out_f = open(save_path, "w")
        for doc_id, offset in todo:
            content: str = ms_reader.get_content(doc_id)
            out_f.write(content + "\n")
        out_f.close()

    total_line = 20000
    skip = True
    ticker = TimeEstimator(total_line, "reading", 1000)
    with open_top100(split) as top100f:
        last_topic_id = None
        cur_doc_ids = set()
        for line_no, line in enumerate(top100f):
            if skip:
                if not line.startswith("532603"):
                    continue
                else:
                    tprint("skip done")
                    remain_lines = total_line - line_no
                    skip = False

            [topic_id, _, doc_id, rank, _, _] = line.split()
            if last_topic_id is None:
                last_topic_id = topic_id
            elif last_topic_id != topic_id:
                pop(last_topic_id, cur_doc_ids)
                last_topic_id = topic_id
                cur_doc_ids = set()

            ticker.tick()
            cur_doc_ids.add(doc_id)
        pop(last_topic_id, cur_doc_ids)

def main():
    collect_doc_per_query("train")



if __name__ == "__main__":
    main()