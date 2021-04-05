import os
from typing import List, Dict, Tuple

from boilerpipe.extract import Extractor

from list_lib import dict_value_map, lmap, flatten, right
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def load_clueweb_docs(ranked_list_path, html_dir_path) -> Dict[str, List[Tuple[str, str]]]:
    rl: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranked_list_path)

    def load_doc(e: TrecRankedListEntry) -> Tuple[str, str]:
        try:
            html_path = os.path.join(html_dir_path, "{}.html".format(e.doc_id))
            content = open(html_path, "r", encoding="utf-8").read()
        except FileNotFoundError as exception:
            print(exception)
            content = ""
        return e.doc_id, content

    def load_doc_for_entries(e: List[TrecRankedListEntry]):
        return lmap(load_doc, e)

    return dict_value_map(load_doc_for_entries, rl)


def load_all_docs_cleaned(ranked_list_path, html_dir_path) -> Dict[str, List[Tuple[str, str]]]:
    docs_grouped = load_clueweb_docs(ranked_list_path, html_dir_path)

    def extract(html: str) -> str:
        if not html:
            return ""
        extractor = Extractor(extractor='ArticleExtractor', html=html)
        core_text = extractor.getText()
        return core_text

    def extract_for_list(doc_list: List[Tuple[str, str]]):
        return list([(doc_id, extract(raw_html)) for doc_id, raw_html in doc_list])

    return dict_value_map(extract_for_list, docs_grouped)


def load_all_docs_text_only(ranked_list_path, html_dir_path) -> List[str]:
    grouped = load_all_docs_cleaned(ranked_list_path, html_dir_path)
    docs: List[str] = right(flatten(grouped.values()))
    return docs


def main():
    rlp = "C:\\work\\Code\\Chair\\output\\clue_counter_arg\\ranked_list.txt"
    html_dir = "C:\\work\\Code\\Chair\\output\\clue_counter_arg\\docs"
    texts = load_all_docs_text_only(rlp, html_dir)
    print("{} texts loaded".format(len(texts)))


if __name__ == "__main__":
    main()
