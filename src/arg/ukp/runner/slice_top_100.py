from typing import Dict, List

from arg.ukp.ukp_q_path import get_ranked_list_save_dir, num_query_file
from base_type import FileName
from cpath import pjoin
from galagos.parse import load_galago_ranked_list
from galagos.query_runs_ids import Q_CONFIG_ID_BM25_UKP
from galagos.types import GalagoDocRankEntry
from misc_lib import TimeEstimator
from sydney_clueweb.clue_path import index_name_list


def work():
    q_config_id = Q_CONFIG_ID_BM25_UKP
    ranked_list_save_root = get_ranked_list_save_dir(q_config_id)
    doc_ids = set()
    ticker = TimeEstimator(num_query_file)
    for i in range(num_query_file):
        file_name = FileName("{}_{}.txt".format(index_name_list[0], str(i)))
        ranked_list_path = pjoin(ranked_list_save_root, file_name)
        rl: Dict[str, List[GalagoDocRankEntry]] = load_galago_ranked_list(ranked_list_path)

        for key, value in rl.items():
            for entry in value[:100]:
                doc_ids.add(entry.doc_id)
        ticker.tick()

    f = open("{}_uniq_100".format(q_config_id), "w")
    for doc_id in doc_ids:
        f.write("{}\n".format(doc_id))
    f.close()


if __name__ == "__main__":
    work()
