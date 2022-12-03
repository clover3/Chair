from typing import List


class SentTokenLabel:
    def __init__(self, qid, labels: List[int]):
        self.qid: str = qid
        self.labels: List[int] = labels

    def to_json(self):
        return {
            'qid': self.qid,
            'labels': self.labels,
        }

    @classmethod
    def from_json(cls, j):
        qid = j['qid']
        labels = j['labels']
        assert type(qid) == str
        for item in labels:
            assert type(item) == int
        return SentTokenLabel(qid, labels)


class SentTokenBPrediction:
    def __init__(self, qid, predictions: List[int]):
        self.qid: str = qid
        self.predictions: List[int] = predictions

    def to_json(self):
        return {
            'qid': self.qid,
            'predictions': self.predictions,
        }

    @classmethod
    def from_json(cls, j):
        qid = j['qid']
        predictions = j['predictions']
        assert type(qid) == str
        for item in predictions:
            assert type(item) == int
        return SentTokenBPrediction(qid, predictions)