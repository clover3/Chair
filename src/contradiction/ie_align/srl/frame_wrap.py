from typing import List, Tuple, NamedTuple

import requests


class IndexToken(NamedTuple):
    word: str
    loc: int

    def get_word(self):
        return self.word


class SRLFrame(NamedTuple):
    frame: List[IndexToken]
    args: List[Tuple[str, List[IndexToken]]]


def get_tag_tail(tag: str):
    return tag[2:]


def allen_srl(sent):
    url = "http://localhost:8131/predict"
    data = {'sentence': sent}
    response = requests.post(url, json=data)
    r = response.json()
    out_frames = parse_response_to_frames(r)
    return out_frames


def parse_response_to_frames(r):
    verbs = r["verbs"]
    words = r["words"]
    out_frames: List[SRLFrame] = []
    for frame in verbs:
        print(frame)
        tags = frame['tags']
        args = []
        cur_tag: str = ""
        cur_tokens = []
        for idx, word in enumerate(words):
            tag = tags[idx]
            cur_token = IndexToken(word, idx)

            if tag.startswith("B-"):
                if cur_tag:
                    assert cur_tag not in args
                    args.append((cur_tag, cur_tokens))
                tag_name = get_tag_tail(tag)
                cur_tag = tag_name
                cur_tokens = [cur_token]
            elif tag.startswith("I-"):
                cur_tokens.append(cur_token)
            elif tag == "O":
                if cur_tag:
                    args.append((cur_tag, cur_tokens))
                cur_tag = ""
                cur_tokens = []
            else:
                assert False

        if cur_tag:
            args.append((cur_tag, cur_tokens))
        print(args)
        verb = None
        for tag, tokens in args:
            if tag == "V":
                verb = tokens

        if verb is None:
            raise ValueError

        out_frames.append(SRLFrame(verb, args))
    return out_frames


def main():
    sent1 = "yeah i i think my favorite restaurant is always been the one closest  you know the closest as long as it's it meets the minimum criteria you know of good food"
    # ARG1 : my favorite restaurant
    sent2 = "My favorite restaurants are always at least a hundred miles away from my house. "
    sent = "at least a hundred miles away from my house"
    sent1 ="If that investor were willing to pay extra for the security of limited downside, she could buy put options with a strike price of $98, which would lock in her profit on the shares at $18, less whatever the options cost."
    sent2 = "THe strike price could be $8"
    sent1 = "Balloon angioplasty has a modest but significant effect on blood pressure and should be considered for patients with atherosclerotic renal artery stenosis and poorly controlled hypertension."
    sent2 = "We found substantial risks but no evidence of a worthwhile clinical benefit from revascularization in patients with atherosclerotic renovascular disease."
    frames = allen_srl(sent2)

    for f in frames:
        print(f.frame)
        for tag, tokens in f.args:
            s = " ".join(map(IndexToken.get_word, tokens))
            print(tag, s)

if __name__ == "__main__":
    main()