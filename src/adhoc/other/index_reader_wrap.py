from abc import ABC, abstractmethod
from typing import Union, List, Tuple

DocID = Union[int, str]


class IndexReaderIF(ABC):
    @abstractmethod
    def get_df(self, term) -> int:
        pass

    @abstractmethod
    def get_dl(self, doc_id) -> int:
        pass

    @abstractmethod
    def get_postings(self, term) -> List[Tuple[DocID, int]]:
        pass


class IndexReaderPython(IndexReaderIF):
    def __init__(self, get_posting_fn, df, dl_d):
        self.get_posting = get_posting_fn
        self.df = df
        self.dl_d = dl_d


    def get_df(self, term) -> int:
        return self.df[term]

    def get_dl(self, doc_id) -> int:
        return self.dl_d[doc_id]

    def get_postings(self, term) -> List[Tuple[DocID, int]]:
        return self.get_posting(term)
