from typing import List, Tuple

from boilerpipe.extract import Extractor

from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from galagos.parse import parse_doc_jsonl_line
from misc_lib import TimeEstimator


def jsonl_to_swtt(line_itr, tokenizer, num_insts=0) -> List[Tuple[str, SegmentwiseTokenizedText]]:
    if num_insts:
        ticker = TimeEstimator(num_insts)

    output = []
    for line in line_itr:
        doc_id, html = parse_doc_jsonl_line(line)
        if num_insts:
            ticker.tick()
        try:
            extractor = Extractor(extractor='ArticleSentencesExtractor', html=html)
            core_text = extractor.getText()
        except Exception as e:
            print("Exception at Extractor")
            print(e)
            continue
        core_text = str(core_text)
        tt: SegmentwiseTokenizedText = SegmentwiseTokenizedText.from_text_list(core_text.split("\n"), tokenizer)
        output.append((doc_id, tt))
    return output

