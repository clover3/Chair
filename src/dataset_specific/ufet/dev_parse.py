from cache import load_list_from_jsonl
from iter_util import load_jsonl
from cpath import output_path, data_path
from misc_lib import path_join
from typing import List, Iterable, Callable, Dict, Tuple, Set, NamedTuple


class UFETEntry(NamedTuple):
    annot_id: str
    left_context_token: List[str]
    right_context_token: List[str]
    mention_span: str
    y_str: List[str]

    @classmethod
    def from_json(cls, j):
        return UFETEntry(
            j['annot_id'],
            j['left_context_token'],
            j['right_context_token'],
            j['mention_span'],
            j['y_str'],
        )

    def get_full_token(self):
        return self.left_context_token + [self.mention_span] + self.right_context_token


def get_jsonl_path(split):
    return path_join(data_path, "ultrafine_acl18", "release", "crowd", "{}.json".format(split))


def load_types():
    path = path_join(data_path, "ultrafine_acl18", "release", "ontology", "types.txt")
    f = open(path, "r")
    output = []
    for line in f:
        output.append(line.strip())
    return output


def load_ufet(split) -> List[UFETEntry]:
    return load_list_from_jsonl(get_jsonl_path(split), UFETEntry.from_json)


def main():
    def assert_str_list(str_list: List[str]):
        assert type(str_list) == list
        try:
            assert type(str_list[0]) == str
        except IndexError:
            pass

    for e in load_ufet("dev"):
        s = " ".join(e.left_context_token) + "[{}]".format(e.mention_span) + " ".join(e.right_context_token)
        print(s)


if __name__ == "__main__":
    main()