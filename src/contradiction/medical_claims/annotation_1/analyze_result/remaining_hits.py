import re

from cache import save_to_pickle, load_from_pickle
from contradiction.medical_claims.mturk.mturk_api_common import ask_hit_id, get_all_available
from tab_print import print_table


def save():
    hit_type_id ='3BHRGUFAIAS5Z7UMKDLPVWH7BFFCSY'
    save_name = "hits_{}".format(hit_type_id)
    hits = ask_hit_id(hit_type_id)
    save_to_pickle(hits, save_name)


def save_all_available():
    hits = get_all_available()
    save_name = "hits_all_available"
    save_to_pickle(hits, save_name)


def main():
    # hit_type_id = '3BHRGUFAIAS5Z7UMKDLPVWH7BFFCSY'
    #
    # save_name = "hits_{}".format(hit_type_id)
    save_name = "hits_all_available"
    hits = load_from_pickle(save_name)

    reg = re.compile(r"https://ecc.neocities.org/(\d+/\d+).html")
    rows = []
    for hit in hits:
        match = reg.search(hit['Question'])
        if match is not None:
            hit_input = match.group(1)
            row = [hit_input,
                  hit['Title'], hit['HITId'],
                  hit['MaxAssignments'], hit['NumberOfAssignmentsAvailable'],
                  hit['NumberOfAssignmentsCompleted'],
                  hit['HITStatus']]
            rows.append(row)
    rows.sort(key=lambda x: x[0])
    print_table(rows)


if __name__ == "__main__":
    save_all_available()
    main()
