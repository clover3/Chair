from typing import Dict, Tuple

from arg.qck.decl import get_format_handler
from arg.qck.prediction_reader import load_combine_info_jsons
from arg.util import load_run_config
from cache import load_pickle_from
from trec.trec_parse import write_trec_relevance_judgement
from trec.types import TrecRelevanceJudgementEntry


def main(info_path, input_type, label_dict_path, save_path):
    f_handler = get_format_handler(input_type)
    info: Dict[str, Dict] = load_combine_info_jsons(info_path, f_handler.get_mapping(), f_handler.drop_kdp())
    label_dict: Dict[Tuple[str, str], bool] = load_pickle_from(label_dict_path)

    l = []
    for entry in info.values():
        key = f_handler.get_pair_id(entry)
        query_id, candidate_id = key
        if key in label_dict:
            correctness = label_dict[key]
        else:
            correctness = False
        e = TrecRelevanceJudgementEntry(query_id, candidate_id, int(correctness))
        l.append(e)

    write_trec_relevance_judgement(l, save_path)


if __name__ == "__main__":
    run_config = load_run_config()
    main(run_config["info_path"],
         run_config["input_type"],
         run_config["label_dict_path"],
         run_config["save_path"])