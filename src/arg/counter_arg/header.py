from typing import NamedTuple, List

topics = [
    "culture",
    "digital-freedoms",
    "economy",
    "education",
    "environment",
    "free-speech-debate",
    "health",
    "international",
    "law",
    "philosophy",
    "politics",
    "religion",
    "science",
    "society",
    "sport"
]

splits = ["training", "validation", "test"]


class ArguDataID(NamedTuple):
    id: str

    def __str__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

    @classmethod
    def from_windows_rel_path(cls, rel_path):
        new_id = rel_path.replace("\\", "/")
        name = new_id.replace("/_con/", "/con/")
        return ArguDataID(id=name)

    @classmethod
    def from_name(cls, name):
        return ArguDataID(id=name)

    @classmethod
    def from_linux_rel_path(cls, rel_path):
        name = rel_path.replace("/_con/", "/con/")
        return ArguDataID(id=name)

    @classmethod
    def from_rel_path(cls, rel_path):
        if "\\" in rel_path:
            return cls.from_windows_rel_path(rel_path)
        else:
            return cls.from_linux_rel_path(rel_path)


class Passage(NamedTuple):
    text: str
    id: ArguDataID

    def __str__(self):
        return self.text


class ArguDataPoint(NamedTuple):
    text1: Passage
    text2: Passage
    annotations: List


num_problems = {'training': 4065, 'validation': 1287, 'test': 1401}
