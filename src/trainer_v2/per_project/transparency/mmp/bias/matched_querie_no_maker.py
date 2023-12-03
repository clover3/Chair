from table_lib import tsv_iter
from trainer_v2.per_project.transparency.misc_common import save_tsv
from trainer_v2.per_project.transparency.mmp.bias.car_model_extract import load_car_maker_related_keywords
from cpath import output_path
from misc_lib import path_join



def iter_match_queries(split):
    for job_no in range(54):
        p = path_join(output_path, "mmp", "bias", f"matched_query_{split}", f"{job_no}.tsv")
        yield from tsv_iter(p)


def main():
    target_terms = load_car_maker_related_keywords()
    qid_query_list = iter_match_queries("dev")

    selected = []
    for qid, query in qid_query_list:
        tokens = query.lower().split()
        skip = False
        for t in tokens:
            if t in target_terms:
                skip = True
                break

        if not skip:
            selected.append((qid, query))

    save_path = path_join(output_path, "mmp", "bias", "dev_matched_car_queries_no_maker.tsv")
    save_tsv(selected, save_path)



if __name__ == "__main__":
    main()