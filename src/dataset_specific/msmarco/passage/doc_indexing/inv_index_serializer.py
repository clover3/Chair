import json
from typing import Iterable, Dict, List, Tuple, Any
import msgpack


def save_inv_index_as_jsonl(inv_index, save_path):
    with open(save_path, "w") as f:
        def generate_json_serializable_items() -> Iterable:
            for term, postings in inv_index.items():
                yield {'term': term}
                for doc_id, cnt in postings:
                    yield (doc_id, cnt)

        itr = generate_json_serializable_items()
        itr = map(json.dumps, itr)
        for line in itr:
            f.write(line + "\n")


def save_inv_index_as_binary(inv_index, save_path):
    with open(save_path, "wb") as f:
        def generate_json_serializable_items() -> Iterable:
            for term, postings in inv_index.items():
                yield {'term': term}
                for doc_id, cnt in postings:
                    yield (doc_id, cnt)

        packer = msgpack.Packer()
        itr = generate_json_serializable_items()
        for data in itr:
            f.write(packer.pack(data))


def save_inv_index_as_binary2(inv_index, save_path):
    with open(save_path, "wb") as f:
        def generate_json_serializable_items() -> Iterable:
            for term, postings in inv_index.items():
                yield (term, postings)

        packer = msgpack.Packer()
        itr = generate_json_serializable_items()
        for data in itr:
            f.write(packer.pack(data))


def load_inv_index_from_binary(save_path):
    with open(save_path, "rb",) as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        itr = unpacker
        inv_index = parse_inv_index(itr)
        return inv_index


def load_inv_index_from_binary2(save_path):
    with open(save_path, "rb",) as f:
        unpacker = msgpack.Unpacker(f, raw=False)
        inv_index = {}
        itr = unpacker
        for term, postings in itr:
            inv_index[term] = postings
        return inv_index



def load_inv_index_from_jsonl(save_path) -> Dict[str, List[Tuple[Any, Any]]]:
    def json_iter():
        for line in open(save_path, "r"):
            yield json.loads(line.strip())

    itr = json_iter()
    inv_index = parse_inv_index(itr)
    return inv_index


def parse_inv_index(itr: Iterable):
    inv_index = {}
    cur_term = None
    cur_posting: List[Tuple[Any, Any]] = []
    for j_item in itr:
        if type(j_item) == dict and len(j_item) == 1:
            term = j_item['term']
            if cur_term is not None:
                inv_index[cur_term] = cur_posting
            cur_posting = []
            cur_term = term
        elif (type(j_item) == tuple or type(j_item) == list) and len(j_item) == 2:
            cur_posting.append(j_item)
        else:
            print("Item is not expected: {}".format(j_item))

    inv_index[cur_term] = cur_posting
    return inv_index