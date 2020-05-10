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


class Passage(NamedTuple):
    text: str
    id: str

    def __str__(self):
        return self.text


class ArguDatapoint(NamedTuple):
    text1: Passage
    text2: Passage
    annotations: List