import json
import pickle
import sys

from boilerpipe.extract import Extractor


def parse_line(line):
    j = json.loads(line, strict=False)
    html = j['content']
    doc_id = j['id']
    extractor = Extractor(extractor='ArticleExtractor', html=html)
    core_text = extractor.getText()
    core_text = str(core_text)
    return {
        'doc_id': doc_id,
        'html': html,
        'core_text': core_text,
    }


def main():
    jsonl_path = sys.argv[1]
    save_path = sys.argv[2]
    f = open(jsonl_path, "r")
    output = []
    for line in f:
        c = parse_line(line)
        output.append(c)

    pickle.dump(output, open(save_path, "wb"))


if __name__ == "__main__":
    main()
