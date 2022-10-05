from contradiction.ists.save_path_helper import get_save_path
from trec.trec_parse import load_ranked_list, write_trec_ranked_list_entry
from trec.types import TrecRankedListEntry


def main():
    run_name = "nlits"
    save_path = get_save_path(run_name)
    ranked_list = load_ranked_list(save_path)

    output_ranked_list = []
    for e in ranked_list:
        new_doc_id = str(int(e.doc_id) + 1)
        new_e = TrecRankedListEntry(e.query_id,
                                    new_doc_id,
                                    e.rank,
                                    e.score,
                                    e.run_name)
        output_ranked_list.append(new_e)

    write_trec_ranked_list_entry(output_ranked_list, save_path)


if __name__ == "__main__":
    main()