import os

from arg.counter_arg_retrieval.build_dataset.run4.data_prep.split_documents import load_run4_swtt
from cpath import output_path
from list_lib import flatten, left
from trec.ranked_list_util import assign_rank
from trec.trec_parse import load_ranked_list_grouped, write_trec_ranked_list_entry


def main():
    doc_ids = left(load_run4_swtt())
    rlg = load_ranked_list_grouped(os.path.join(output_path, "ca_building", "run4", "pc_res.filtered.txt"))
    save_path = os.path.join(output_path, "ca_building", "run4", "pc_res.filtered2.txt")
    new_rlg = {}
    for key, values in rlg.items():
        new_entries = []
        for e in values:
            if e.doc_id in doc_ids:
                new_entries.append(e)
            else:
                print(e.doc_id, 'not found')

        new_rlg[key] = assign_rank(new_entries)

    write_trec_ranked_list_entry(
        flatten(new_rlg.values()), save_path)


if __name__ == "__main__":
    main()
