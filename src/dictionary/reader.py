from collections import Counter
import numpy as np

####
#  Reading webster dictionary
#
#  Assumption : First line of the entry is capitalized word
#  Header comes after the word without empty line
#  Content follows Header ( empty line may or may not exists )
#
#  Content contains empty line

def is_empty(line):
    return len(line.strip()) == 0


def is_numbering(line):
    if len(line) < 3:
        return False
    if line[0].isnumeric() and line[1] == ".":
        return True

    if line[0] == "(" and line[1].isalpha() and line[2] == ")":
        return True
    else:
        return False


class DictionaryReader:
    def __init__(self, entries):
        self.entries = entries
        # entry = {
        #         'word': word,
        #         'content': content_list,
        #         'head': head_list,
        #         }

    @classmethod
    def open(cls, path):
        STATE_BEGIN = 0
        STATE_HEAD = 1
        STATE_CONTENT_BEGIN = 2
        STATE_CONTENT_MIDDLE = 3
        # STATE_HEAD -> STATE_CONTENT_BEGIN
        # STATE_CONTENT_BEGIN -> STATE_CONTENT_MIDDLE
        #                     -> STATE_HEAD
        # STATE_CONTENT_MIDDLE -> STATE_CONTENT_BEGIN

        f = open(path, "r", encoding="utf-8")


        WORD_TYPE_SINGLE = 0
        WORD_TYPE_PHRASAL = 1
        WORD_TYPE_MULTI_SURFACE = 2
        def parse_word_line(word_line):
            if len(word_line.split()) == 1:
                return word_line, WORD_TYPE_SINGLE

            if ";" in word_line:
                words = word_line.split(";")
                words = list([w.strip() for w in words])
                return words, WORD_TYPE_MULTI_SURFACE
            else:
                return word_line, WORD_TYPE_PHRASAL



        def is_new_word_entry(line):
            if not line.isupper():
                return False
            if "  "  in line:
                return False
            if line[0].isnumeric():
                return False

            return True



        line_no = 0
        cnt_no_head = 0
        cnt_no_content = 0
        data = []
        state = STATE_BEGIN
        for line in f:
            line = line.strip()
            if state == STATE_BEGIN:
                word_line = line
                head_list = []
                content_list = []
                state = STATE_HEAD

            elif state == STATE_HEAD:
                if is_empty(line):  # empty line
                    state = STATE_CONTENT_BEGIN
                elif line.startswith("Defn:") or is_numbering(line):
                    state = STATE_CONTENT_BEGIN
                    state = STATE_CONTENT_MIDDLE
                    content_list.append(line)
                else:
                    head_list.append(line)

            elif state == STATE_CONTENT_BEGIN:
                if is_new_word_entry(line):
                    # Pop
                    assert word_line is not None
                    if head_list == []:
                        print("WARNING : no head found : {}".format(word_line))
                        cnt_no_head += 1
                    if content_list == [] :
                        print("WARNING : no content found : {} . line_no = {}".format(word_line, line_no))
                        cnt_no_content += 1
                        if line_no not in [505961, 596938, 967251]:
                            raise Exception("")

                    words, w_type = parse_word_line(word_line)
                    if w_type == WORD_TYPE_MULTI_SURFACE:
                        pass
                    else:
                        words = [words]

                    for w in words:
                        entry = {
                            'word': w,
                            'content': content_list,
                            'head': head_list,
                            'type':w_type
                        }
                        data.append(entry)


                    word_line = line
                    head_list = []
                    content_list = []
                    state = STATE_HEAD
                else:
                    content_list.append(line)
                    if not is_empty(line):
                        state = STATE_CONTENT_MIDDLE


            elif state == STATE_CONTENT_MIDDLE:
                content_list.append(line)
                if is_empty(line):  # empty line
                    state = STATE_CONTENT_BEGIN

            else:
                print("STATE: ", state)
                print(line)
                raise Exception("Unexpected state during parsing")

            line_no += 1

        print("# of entry : {}".format(len(data)))
        print("# of entry w/o head {}".format(cnt_no_head))
        return DictionaryReader(data)

list_pos = ['n.', 'n. pl.', 'n. sing.', 'n. sing. & pl.',
            'v.', 'v. t.', 'v. i.','v. t. & i.',
            'a.', 'adv.',
            'prep.', 'conj.', 'pron.', 'interj.',
            'p. pr.',
            'obs.',
            'p. p.', 'imp.']

def contain_any_pos(str):
    for pos in list_pos:
        if pos in str:
            return True
    return False


def detect_pos(str):
    matched = []
    for pos in list_pos:
        idx = str.find(pos)
        if idx > 0:
            if str[idx-1] in [" ", "&"]:
                matched.append(pos)
        elif idx == 0 :
            matched.append(pos)
    return matched

def split_paragraphs(lines):
    paragraphs = []
    cursor = 0
    begin = 0
    while cursor < len(lines):
        if is_empty(lines[cursor]):
            para = (" ".join(lines[begin:cursor])).strip()
            if len(para) > 0 :
                paragraphs.append(para)
            begin = cursor + 1
        cursor += 1
    return paragraphs




class DictionaryParser:
    def __init__(self, dictionary_reader:DictionaryReader):
        def parse_head(word, head_lines):

            def is_pronunciation(token):
                score =0
                target = token
                if token[-1] ==",":
                    score += 100
                    target = target[:-1]

                if token.lower() == word.lower():
                    score += 1000

                l = 0
                match = 0
                for c in token:
                    if c.isalpha() :
                        l += 1
                        if c.lower() in word.lower():
                            match += 1

                if l > 0 :
                    if match/l > 0.5:
                        score += 100
                    elif match/l > 0.3 and len(token) > 3:
                        score += 100
                else:
                    score -= 100
                l_diff = len(token) - len(word)

                upper_cut = len(token) * 0.5
                if l_diff > upper_cut:
                    score -= 10 * (l_diff-upper_cut)
                elif l_diff < -3 :
                    score -= 10 * (abs(l_diff) - 2)

                if score > 20:
                    return True
                else:
                    return False

            if head_lines == []:
                return {}
            # 1st line usually come in
            # [Pronunciation / POS / Etymology
            first_line = head_lines[0]
            first_comma = first_line.find(",")
            first_space = first_line.find(" ")

            first_token = head_lines[0].split()[0]
            if is_pronunciation(first_token):
                pronunciation = first_token
                pos_search_begin = len(first_token)
            else:
                pronunciation = None
                pos_search_begin = 0

            end_features = ["Etym:", "["]

            end_idx = [len(first_line)]
            for key in end_features:
                idx = first_line.find(key, pos_search_begin)
                if idx > 0:
                    end_idx.append(idx)

            pos_search_end = min(end_idx)
            pos_hypo = first_line[pos_search_begin:pos_search_end].strip()
            pos_list = detect_pos(pos_hypo)

            cursor = pos_search_end
            head_remains = first_line[cursor:] + " ".join(head_lines[1:])
            return {
                "pronunciation": pronunciation,
                "pos": pos_list,
                "head_remain": head_remains,
            }

        def parse_content(content_lines):
            #STYLE_SINGLE_DEF = 0
            STYLE_NUMBER_DEF = 1
            STYLE_ALPHA_DEF = 2
            STYLE_UKNOWN = -1

            observed_style = STYLE_UKNOWN

            idx = 0

            def is_num_head(line):
                return len(line) > 1 and line[0].isnumeric() and line[1] == "."

            def is_alpha_head(line):
                return len(line) > 3 and line[0] == "(" and line[1].isalpha() and line[2] == ")"

            def is_def_head(line):
                return line.startswith("Defn:")

            def is_new_heading(line, observed_style, prev_line):
                if is_num_head(line):
                    return True, STYLE_NUMBER_DEF
                elif observed_style != STYLE_NUMBER_DEF and is_alpha_head(line):
                    return True, STYLE_ALPHA_DEF
                elif observed_style == STYLE_UKNOWN and is_def_head(line):
                    return True, STYLE_UKNOWN
                else:
                    return False, STYLE_UKNOWN

            entries = []
            try:
                cur_entry = []
                while idx < len(content_lines):
                    line = content_lines[idx]
                    prev_line = content_lines[idx-1] if idx > 0 else None
                    f_new_head, style = is_new_heading(line, observed_style, prev_line)

                    if f_new_head:
                        observed_style = style
                        if observed_style in [STYLE_ALPHA_DEF, STYLE_NUMBER_DEF] \
                            and len(("".join(cur_entry)).strip()) > 1:
                            entries.append(cur_entry)
                            cur_entry = []
                    cur_entry.append(line)

                    idx += 1
            except Exception as e:
                print("Word", word)
                print("Current Line:")
                print(line)
                for line in content_lines:
                    print(line)
                raise e

            if cur_entry and "".join(cur_entry).strip():
                entries.append(cur_entry)

            wrong = False
            for e in entries:
                if len("".join(e)) < 10:
                    wrong = True

            if wrong:
                print(word)
                for e in entries:
                    print(e)




            return entries

        missing_count = Counter()
        pos_count = Counter()

        self.word2entry = {}
        for entry in dictionary_reader.entries:
            word = entry['word']
            content = parse_content(entry['content'])
            head = parse_head(word, entry['head'])
            if word not in self.word2entry:
                self.word2entry[word] =[]
            self.word2entry[word].append((head, content))
            if entry['head'] == []:
                missing_count['head'] += 1
            else:
                if not head['pronunciation']:
                    missing_count['pronunciation'] += 1
                if not head['head_remain'] or "Etym:" not in head['head_remain']:
                    missing_count['etym'] += 1
                if not head['pos']:
                    missing_count['pos'] += 1
                for pos in head['pos']:
                    pos_count[pos] += 1

        for key in missing_count.keys():
            print("Missing {} : {}".format(key, missing_count[key]))
        for item, cnt in pos_count.most_common():
            print(item, cnt)

def simple_parser(d1):
    d = {}
    for e in d1.entries:
        w = e['word']
        text = " ".join(e['content'])
        if w not in d:
            d[w] = text
        else:
            d[w] += " " + text

    return d


def simple_parser_demo():
    path = "c:\\work\\Data\\webster\\webster_headerless.txt"
    d1 = DictionaryReader.open(path)
    #d2 = DictionaryParser(d1)
    d = simple_parser(d1)
    l_list = []
    for w in d:
        l_list.append(len(d[w]))

    line_cnt =0
    for w in d:
        print(w, end="\t")
        line_cnt += 1
        if line_cnt == 20:
            line_cnt = 0
            print()

    print("avg", np.average(l_list))
    print("std", np.std(l_list))

def parser_dev():
    path = "c:\\work\\Data\\webster\\webster_headerless.txt"
    d1 = DictionaryReader.open(path)
    d2 = DictionaryParser(d1)

def dictionary_reader_demo():
    path = "c:\\work\\Data\\webster\\webster_headerless.txt"
    d1 = DictionaryReader.open(path)
    for e in d1.entries:
        w = e['word']
        print(w)
        print(e['content'])



if __name__ == "__main__":
    dictionary_reader_demo()