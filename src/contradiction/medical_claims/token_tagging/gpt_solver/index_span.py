from typing import NamedTuple, List


def strip_index(text, st, ed):
    st_cursor = st
    while text[st_cursor].isspace() and st_cursor < ed:
        st_cursor += 1

    ed_cursor = ed
    while text[ed_cursor - 1].isspace() and ed_cursor - 1 > st_cursor:
        ed_cursor -= 1
    return st_cursor, ed_cursor


def strip_char(text, st, ed, ch):
    st_cursor = st
    while text[st_cursor] == ch and st_cursor < ed:
        st_cursor += 1

    ed_cursor = ed
    while text[ed_cursor - 1] == ch and ed_cursor - 1 > st_cursor:
        ed_cursor -= 1
    return st_cursor, ed_cursor


def strip_char_set(text, st, ed, ch_set):
    st_cursor = st
    while st_cursor < ed and text[st_cursor] in ch_set:
        st_cursor += 1

    ed_cursor = ed
    while ed_cursor - 1 > st_cursor and text[ed_cursor - 1] in ch_set:
        ed_cursor -= 1
    return st_cursor, ed_cursor


class IndexedSpan(NamedTuple):
    src_text: str
    st: int
    ed: int

    def strip(self):
        st_new, ed_new = strip_index(self.src_text, self.st, self.ed)
        return IndexedSpan(self.src_text, st_new, ed_new)

    def strip_quotation(self):
        st_new, ed_new = strip_char(self.src_text, self.st, self.ed, "\"")
        return IndexedSpan(self.src_text, st_new, ed_new)

    def split(self, delimiter=None):
        if delimiter is not None and delimiter:
            return split_indexed(self.src_text, self.st, self.ed, delimiter)
        else:
            return space_split(self.src_text, self.st, self.ed)

    @classmethod
    def from_text(cls, src_text: str, pattern: str):
        i = src_text.find(pattern)
        if i < 0:
            raise IndexError
        return IndexedSpan(src_text, i, i+len(pattern))

    def to_text(self):
        return self.src_text[self.st: self.ed]

    def get_sp_token_indices(self) -> List[int]:
        token_idx = 0
        in_token = False
        output = []
        for i, t in enumerate(self.src_text):
            if i >= self.ed:
                break

            if in_token:
                if t.isspace():
                    token_idx += 1
                    in_token = False
            else:
                if t.isspace():
                    pass
                else:  # New token starts
                    in_token = True
                    if self.st <= i < self.ed:
                        output.append(token_idx)

        return output

    def __str__(self):
        return "IndexedSpan(display_text=\"{}\", st={}, ed={})".format(self.to_text(), self.st, self.ed)


def space_split(full_text, st, ed) -> List[IndexedSpan]:
    output: List[IndexedSpan] = []
    token_st = st
    in_token = False
    for i in range(st, ed):
        if in_token:
            if full_text[i].isspace():
                in_token = False
                output.append(IndexedSpan(full_text, token_st, i))
            else:
                pass
        else:
            if full_text[i].isspace():
                pass
            else:
                in_token = True
                token_st = i
    if in_token:
        output.append(IndexedSpan(full_text, token_st, ed))

    return output


def split_indexed(full_text, st, ed, delimiter) -> List[IndexedSpan]:
    output: List[IndexedSpan] = []
    cursor = st
    while cursor < ed:
        new_point = full_text.find(delimiter, cursor, ed)
        if new_point >= 0:
            output.append(IndexedSpan(full_text, cursor, new_point))
            cursor = new_point + len(delimiter)
        else:
            output.append(IndexedSpan(full_text, cursor, ed))
            break
    return output


def find_all(text, pattern):
    text_l = text.lower()
    pattern_l = pattern.lower()
    output = []
    cursor = 0
    while cursor < len(text):
        i = text_l.find(pattern_l, cursor)
        cursor = i + 1
        if i >= 0:
            output.append(i)
        else:
            break

    return output


def find_all_as_index_span(text, pattern) -> List[IndexedSpan]:
    output: List[IndexedSpan] = []
    for i in find_all(text, pattern):
        output.append(IndexedSpan(text, i, i+len(pattern)))
    return output


def dev():
    claim1 = "Supplementation during pregnancy with a medical food containing L-arginine and antioxidant vitamins reduced the incidence of pre-eclampsia in a population at high risk of the condition."

    for span in space_split(claim1, 0, len(claim1)):
        print(span.to_text(), span.st, span.ed)



def main():
    dev()


if __name__ == "__main__":
    main()