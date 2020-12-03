import json
from typing import List, Dict

from data_generator.tokenizer_wo_tf import get_tokenizer


def clean_content(content: str) -> str:
    content = content.strip()
    s1 = "<TEXT>"
    if content.startswith(s1):
        content = content[len(s1):]

    s2 = "</TEXT>"
    if content.endswith(s2):
        content = content[:-len(s2)]

    return content


def jsonl_tokenize(jsonl_path) -> Dict[str, List[str]]:
    tokenizer = get_tokenizer()
    # load jsonl
    # pre-process (remove structural stuff)
    # save_dictionary
    out_dict: Dict[str, List[str]] = dict()

    for idx, line in enumerate(open(jsonl_path, "r")):
        j = json.loads(line)
        doc_id = j['id']
        content = j['content']
        content_cleaned = clean_content(content)
        tokens = tokenizer.tokenize(content_cleaned)
        out_dict[doc_id] = tokens

    return out_dict
