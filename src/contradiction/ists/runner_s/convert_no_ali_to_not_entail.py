import re

from contradiction.ists.save_path_helper import get_save_path, get_not_entail_save_path
from trec.trec_parse import load_ranked_list, write_trec_ranked_list_entry
from trec.types import TrecRankedListEntry


def main(run_name):
    genre = "headlines"
    split = "train"
    src_path = get_save_path(run_name)
    entries = load_ranked_list(src_path)
    def convert_entry(e: TrecRankedListEntry):
        def convert_qid(qid):
            regex = r"noali_(\d+)_(\d+)"
            m = re.match(regex, qid)
            return "{}-{}".format(m.group(1), m.group(2))
        new_qid = convert_qid(e.query_id)
        return TrecRankedListEntry(new_qid, e.doc_id, e.rank, e.score, e.run_name)

    out_entries = list(map(convert_entry, entries))
    save_path = get_not_entail_save_path(genre, split, run_name)
    write_trec_ranked_list_entry(out_entries, save_path)



if __name__ == "__main__":
    run_name_list = ["exact_match", "idf", "nlits", "nlits_punc", "word2vec"]
    run_name_list = ["partial_seg"]
    for run_name in run_name_list:
        main(run_name)