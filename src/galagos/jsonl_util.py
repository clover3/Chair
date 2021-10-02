import json
from typing import Dict


def load_jsonl(jsonl_path) -> Dict[str, str]:
    docs_d = {}
    for line_no, line in enumerate(open(jsonl_path, "r", newline="\n")):
        try:
            j = json.loads(line, strict=False)
            docs_d[j['id']] = j['content']
        except json.decoder.JSONDecodeError:
            print(line)
            print("json.decoder.JSONDecodeError", line_no)
    return docs_d