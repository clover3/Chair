import json
import sys

from arg.qck.decl import QCKQuery, QCKCandidate, KnowledgeDocumentPart


def convert(info):
    new_info = {
        'query': QCKQuery(str(info['qid']), ""),
        'candidate': QCKCandidate(str(info['cid']), ""),
        'kdp': KnowledgeDocumentPart("", 0, 0, []),
    }
    return new_info


def main(file_path, out_file_path):
    j = json.load(open(file_path, "r", encoding="utf-8"))
    new_j = {}
    for data_id, info in j.items():
        new_j[data_id] = convert(info)
    json.dump(new_j, open(out_file_path, "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
